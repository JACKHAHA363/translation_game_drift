""" Trainer objects for tinetune
"""
import os
import torch
import time

from ld_research.settings import LOGGER, FR, EN, DE
from ld_research.text import Multi30KLoader, Multi30KDataset
from ld_research.model import Agent, ValueNetwork
from ld_research.training.optimizers import Optimizer
from ld_research.training.utils import NMTLoss, StatisticsReport, process_batch_update_stats
from ld_research.training.base import BaseTrainer


class AgentType:
    FR_EN = 'fr_en'
    EN_DE = 'en_de'
    VALUE = 'value'


ALL_TYPES = [AgentType.FR_EN, AgentType.EN_DE, AgentType.VALUE]
TRANSLATOR_TYPES = [AgentType.FR_EN, AgentType.EN_DE]


class Trainer(BaseTrainer):
    """ Fine tuner """
    def __init__(self, opt):
        """ Constructor """
        super(Trainer, self).__init__(opt)
        ckpt = self.get_latest_checkpoint()
        self._build_data_loaders()
        self._build_models(ckpt)
        self._build_optimizers(ckpt)
        self.de_criterion = NMTLoss(opt.label_smoothing,
                                    tgt_vocab_size=len(self.de_vocab),
                                    device=self.en_de_agent.device)
        self.en_criterion = NMTLoss(opt.label_smoothing,
                                    tgt_vocab_size=len(self.en_vocab),
                                    device=self.fr_en_agent.device)

    """
    Main method
    """
    def start_training(self):
        """ Start training """
        self.fr_en_agent.train()
        self.en_de_agent.train()
        self.value_net.train()
        step = self.fr_en_optimizer.curr_step + 1
        de_train_stats = StatisticsReport()
        en_train_stats = StatisticsReport()
        train_start = time.time()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):
                batch.to(device=self.device)

                # Take action and translate
                trans_en, trans_en_lengths = self._get_en_translation(batch)
                action_logprobs, actions, action_masks = self.fr_en_agent(src=batch.fr, src_lengths=batch.fr_lengths,
                                                                          tgt=trans_en, tgt_lengths=trans_en_lengths)
                values, = self.value_net(src=batch.fr, src_lengths=batch.fr_lengths,
                                         tgt=trans_en, tgt_lengths=trans_en_lengths)

                # Get Germany Loss
                de_batch_results = process_batch_update_stats(src=trans_en[:, 1:],
                                                              src_lengths=trans_en_lengths - 1,
                                                              tgt=batch.de, tgt_lengths=batch.de_lengths,
                                                              stats_rpt=de_train_stats, agent=self.en_de_agent,
                                                              criterion=self.de_criterion)

                # Train en_de_agent
                en_de_loss = de_batch_results[1]
                avg_en_de_loss = en_de_loss / de_batch_results[0].n_words
                self.en_de_optimizer.zero_grad()
                avg_en_de_loss.backward()
                self.en_de_optimizer.step()

                # Train fr_en_agent
                # [bsz]
                rewards = self.get_rewards(de_batch_results)

                # reinforce
                # [bsz, seq_len]
                adv = rewards.unsqueeze(-1) - values
                pg_loss, value_loss, ent_loss = self.get_rl_loss(adv=adv,
                                                                 logprobs=action_logprobs,
                                                                 actions=actions,
                                                                 action_masks=action_masks)
                fr_en_loss = pg_loss + self.opt.v_coeff * value_loss + self.opt.ent_coeff * ent_loss
                self.value_optimizer.zero_grad()
                self.fr_en_optimizer.zero_grad()
                fr_en_loss.backward()
                self.value_optimizer.step()
                self.fr_en_optimizer.step()

                # Evaluate fr_en agent in training
                process_batch_update_stats(src=batch.fr, src_lengths=batch.fr_lengths,
                                           tgt=batch.en, tgt_lengths=batch.en_lengths,
                                           stats_rpt=en_train_stats, agent=self.fr_en_agent,
                                           criterion=self.en_criterion)

                # Validate
                if (step + 1) % self.opt.valid_steps == 0:
                    with torch.no_grad():
                        self.validate(step=step)
                        self.fr_en_agent.train()
                        self.en_de_agent.train()
                        self.value_net.train()

                # Logging and Saving
                en_train_stats = self._report_training(step, en_train_stats, train_start,
                                                       prefix='train/en',
                                                       learning_rate=self.fr_en_optimizer.learning_rate,
                                                       rewards=rewards)
                de_train_stats = self._report_training(step, de_train_stats, train_start,
                                                       prefix='train/de',
                                                       learning_rate=self.en_de_optimizer.learning_rate)

                # Increment step
                step += 1

    def validate(self, step):
        """ Validatation """
        self.fr_en_agent.eval()
        self.en_de_agent.eval()
        self.value_net.eval()
        de_valid_stats = StatisticsReport()
        en_valid_stats = StatisticsReport()

        # For bleu
        en_actuals = []
        en_references = []
        de_actuals = []
        de_references = []

        # Start main loop
        for batch in self.valid_loader:
            batch.to(device=self.device)

            # Get English stats
            process_batch_update_stats(src=batch.fr, src_lengths=batch.fr_lengths,
                                       tgt=batch.en, tgt_lengths=batch.en_lengths,
                                       stats_rpt=en_valid_stats, agent=self.fr_en_agent,
                                       criterion=self.en_criterion)

            # Get translated English
            trans_en, trans_en_lengths = self.fr_en_agent.batch_translate(src=batch.fr,
                                                                          src_lengths=batch.fr_lengths,
                                                                          max_lengths=batch.fr_lengths)
            en_actuals += self.en_vocab.to_sentences(ids=trans_en.tolist())
            en_references += [[en_sent] for en_sent in self.en_vocab.to_sentences(ids=batch.en.tolist())]

            # Get Germany stats
            process_batch_update_stats(src=trans_en[:, 1:],
                                       src_lengths=trans_en_lengths - 1,
                                       tgt=batch.de, tgt_lengths=batch.de_lengths,
                                       stats_rpt=de_valid_stats, agent=self.en_de_agent,
                                       criterion=self.de_criterion)

            # Get translated Germany
            trans_de, trans_de_lengths = self.en_de_agent.batch_translate(src=trans_en[:, 1:],
                                                                          src_lengths=trans_en_lengths - 1,
                                                                          max_lengths=100)
            de_actuals += self.de_vocab.to_sentences(ids=trans_de.tolist())
            de_references += [[de_sent] for de_sent in self.de_vocab.to_sentences(ids=batch.de.tolist())]

        # Reporting
        LOGGER.info('Eng Validation perplexity: %g' % en_valid_stats.ppl())
        LOGGER.info('Eng Validation accuracy: %g' % en_valid_stats.accuracy())
        LOGGER.info('Ger Validation perplexity: %g' % de_valid_stats.ppl())
        LOGGER.info('Ger Validation accuracy: %g' % de_valid_stats.accuracy())
        en_valid_stats.log_tensorboard(prefix='valid/en',
                                       learning_rate=self.fr_en_optimizer.learning_rate,
                                       step=step,
                                       writer=self.writer)
        de_valid_stats.log_tensorboard(prefix='valid/de',
                                       learning_rate=self.en_de_optimizer.learning_rate,
                                       step=step,
                                       writer=self.writer)

        # Bleu Score
        en_valid_stats.report_bleu_score(en_references, en_actuals, self.writer,
                                         prefix='valid/en', step=step)
        de_valid_stats.report_bleu_score(de_references, de_actuals, self.writer,
                                         prefix='valid/de', step=step)

    def checkpoint(self, step):
        """ Maybe do the checkpoint of model """
        if (step + 1) % self.opt.checkpoint_steps == 0:
            LOGGER.info('Checkpoint step {}...'.format(step))
            checkpoint = {agent_type: {'agent_state_dict': self.agents[agent_type].state_dict(),
                                       'optimizer': self.optimizers[agent_type]}
                          for agent_type in ALL_TYPES}
            ckpt_name = 'checkpoint.{}.pt'.format(step)
            ckpt_path = os.path.join(self.opt.save_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

    @staticmethod
    def get_rewards(de_batch_results):
        """ Compute the reward.
            :return rewards: [bsz]
        """
        loss, masks = de_batch_results[2], de_batch_results[5]
        batch_loss = torch.sum(loss * masks,  dim=-1).detach()
        return -batch_loss

    @staticmethod
    def get_rl_loss(adv, logprobs, actions, action_masks):
        """ Return the average policy gradient loss per action
            :param adv: [bsz, seq_len]
            :param logprobs: [bsz, seq_len, nb_actions]
            :param actions: [bsz, seq_len]
            :param action_masks: [bsz, seq_len]
            :return pg_loss, value_loss, ent_loss
        """
        # Get number of actions
        nb_actions = torch.sum(action_masks).item()

        # index get policy log probs
        # [bsz, seq_len]
        logprobs_f = logprobs.view(-1, logprobs.size(2))
        actions_f = actions.contiguous().view(-1, 1)
        indexed_logprobs = logprobs_f.gather(1, actions_f).squeeze(-1)
        policy_logprobs = indexed_logprobs.view(adv.size(0), adv.size(1))

        # Get reinforce loss
        pg_loss = -torch.sum(policy_logprobs * adv.detach() * action_masks) / nb_actions

        # Get value loss
        value_loss = torch.sum((adv * action_masks).pow(2)) / nb_actions

        # Get ent loss
        ent = torch.sum(logprobs * logprobs.exp(), dim=-1)  # [bsz, seq_len]
        ent_loss = -torch.sum(ent * action_masks) / nb_actions

        return pg_loss, value_loss, ent_loss

    """
    Private method
    """

    def _get_en_translation(self, batch):
        """ Return (trans_en, trans_en_lengths """
        self.fr_en_agent.eval()
        trans_en, trans_en_lengths = self.fr_en_agent.batch_translate(src=batch.fr,
                                                                      src_lengths=batch.fr_lengths,
                                                                      max_lengths=batch.fr_lengths)
        self.fr_en_agent.train()
        return trans_en, trans_en_lengths

    def _report_training(self, step, train_stats, train_start, prefix, learning_rate, rewards=None):
        """ Report the training """
        if (step + 1) % self.opt.logging_steps == 0:
            train_stats.output(step=step,
                               learning_rate=learning_rate,
                               num_steps=self.opt.train_steps,
                               train_start=train_start,
                               prefix=prefix)
            train_stats.log_tensorboard(prefix=prefix, writer=self.writer,
                                        learning_rate=learning_rate,
                                        step=step)
            if rewards is not None:
                avg_rewards = torch.mean(rewards).item()
                LOGGER.info("{} Step {}/{}; rewards: {}".format(prefix, step,
                                                                self.opt.train_steps,
                                                                avg_rewards))
                self.writer.add_scalar(prefix + '/rewards', avg_rewards, step)
            train_stats = StatisticsReport()
        return train_stats

    def _build_data_loaders(self):
        """ Build datasets """

        if self.opt.debug:
            LOGGER.info('Debug mode, overfit on test data...')
            train_set = Multi30KDataset('test')
        else:
            train_set = Multi30KDataset('train')
        valid_set = Multi30KDataset('valid')
        self.train_loader = Multi30KLoader(train_set, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=1)
        self.valid_loader = Multi30KLoader(valid_set, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=1)
        self.vocabs = valid_set.vocabs

    def _build_models(self, ckpt=None):
        """ Build all the agents """
        self.agents = dict()
        self.agents[AgentType.VALUE] = ValueNetwork(src_vocab=self.fr_vocab,
                                                    tgt_vocab=self.en_vocab,
                                                    opt=self.opt)
        self.agents[AgentType.FR_EN] = Agent(src_vocab=self.fr_vocab,
                                             tgt_vocab=self.en_vocab,
                                             opt=self.opt)
        self.agents[AgentType.EN_DE] = Agent(src_vocab=self.en_vocab,
                                             tgt_vocab=self.de_vocab,
                                             opt=self.opt)
        if ckpt:
            for agent_type in ALL_TYPES:
                self.agents[agent_type].load_state_dict(ckpt[agent_type]['agent_state_dict'])

        # Try load from pretrain
        else:
            self._initialize_agent(self.agents[AgentType.FR_EN],
                                   self.opt.pretrain_fr_en_ckpt)
            self._initialize_agent(self.agents[AgentType.FR_EN],
                                   self.opt.pretrain_fr_en_ckpt)
            self._initialize_agent(self.agents[AgentType.VALUE])

        for agent_type in ALL_TYPES:
            self.agents[agent_type].to(device=self.device)

    def _build_optimizers(self, ckpt=None):
        """ Build Optimizer """
        save_dicts = {agent_type: None for agent_type in ALL_TYPES}
        if ckpt:
            LOGGER.info('Loading the optimizer info from {}...'.format(self.opt.save_dir))
            self.optimizers = {agent_type: ckpt[agent_type]['optimizer'] for agent_type in ALL_TYPES}
            for agent_type in ALL_TYPES:
                save_dicts[agent_type] = self.optimizers[agent_type].state_dict()
        else:
            opt = self.opt
            self.optimizers = dict()
            self.optimizers[AgentType.VALUE] = Optimizer(
                opt.value_optim, opt.value_learning_rate,
                opt.value_max_grad_norm,
                lr_decay=opt.value_learning_rate_decay,
                start_decay_steps=opt.value_start_decay_steps,
                decay_steps=opt.value_decay_steps,
                beta1=opt.value_adam_beta1,
                beta2=opt.value_adam_beta2,
                adagrad_accum=opt.value_adagrad_accumulator_init,
                decay_method=opt.value_decay_method,
                warmup_steps=opt.value_warmup_steps)
            self.optimizers[AgentType.FR_EN] = Optimizer(
                opt.fr_en_optim, opt.fr_en_learning_rate,
                opt.fr_en_max_grad_norm,
                lr_decay=opt.fr_en_learning_rate_decay,
                start_decay_steps=opt.fr_en_start_decay_steps,
                decay_steps=opt.fr_en_decay_steps,
                beta1=opt.fr_en_adam_beta1,
                beta2=opt.fr_en_adam_beta2,
                adagrad_accum=opt.fr_en_adagrad_accumulator_init,
                decay_method=opt.fr_en_decay_method,
                warmup_steps=opt.fr_en_warmup_steps)
            self.optimizers[AgentType.EN_DE] = Optimizer(
                opt.en_de_optim, opt.en_de_learning_rate,
                opt.en_de_max_grad_norm,
                lr_decay=opt.en_de_learning_rate_decay,
                start_decay_steps=opt.en_de_start_decay_steps,
                decay_steps=opt.en_de_decay_steps,
                beta1=opt.en_de_adam_beta1,
                beta2=opt.en_de_adam_beta2,
                adagrad_accum=opt.en_de_adagrad_accumulator_init,
                decay_method=opt.en_de_decay_method,
                warmup_steps=opt.en_de_warmup_steps)

        # Set parameters by initialize new torch optim inside
        for agent_type in ALL_TYPES:
            self.optimizers[agent_type].set_parameters(params=self.agents[agent_type].named_parameters())

        # Set the states
        if ckpt:
            for agent_type in ALL_TYPES:
                self.optimizers[agent_type].load_state_dict(save_dicts[agent_type])

    def _initialize_agent(self, agent, pretrain_ckpt_path=None):
        """ Intialize agent from pretrain or random """
        if pretrain_ckpt_path:
            ckpt = torch.load(pretrain_ckpt_path,
                              map_location=lambda storage, loc: storage)
            agent.load_state_dict(ckpt['agent'])
        else:
            agent.initialize(self.opt.param_init)

    """
    Properties
    """
    @property
    def en_vocab(self):
        """ Getter """
        return self.vocabs[EN]

    @property
    def fr_vocab(self):
        """ Getter """
        return self.vocabs[FR]

    @property
    def de_vocab(self):
        """ Getter """
        return self.vocabs[DE]

    @property
    def fr_en_agent(self):
        """ Getter """
        return self.agents[AgentType.FR_EN]

    @property
    def value_net(self):
        """ Getter """
        return self.agents[AgentType.VALUE]

    @property
    def en_de_agent(self):
        """ Getter """
        return self.agents[AgentType.EN_DE]

    @property
    def fr_en_optimizer(self):
        """ Getter """
        return self.optimizers[AgentType.FR_EN]

    @property
    def en_de_optimizer(self):
        """ Getter """
        return self.optimizers[AgentType.EN_DE]

    @property
    def value_optimizer(self):
        """ Getter """
        return self.optimizers[AgentType.VALUE]

