""" Run Finetuning with PPO
"""
import torch
import time
from collections import namedtuple
import random

from ld_research.training.utils import StatisticsReport, process_batch_update_stats
from ld_research.training.finetune import Trainer as PGTrainer


class PPOBatch(namedtuple('PPOBatch', ('old_logprobs',
                                       'returns',
                                       'advs',
                                       'batch_idx'))):
    """ A data structure to hold the information need to train each PPO batch """
    pass


class Trainer(PGTrainer):
    """ Use PPO to train """

    def start_training(self):
        """ Start training """
        self.fr_en_agent.train()
        self.en_de_agent.train()
        self.value_net.train()
        step = self.fr_en_optimizer.curr_step
        de_train_stats = StatisticsReport()
        en_train_stats = StatisticsReport()
        train_start = time.time()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):

                # Validate
                if step % self.opt.valid_steps == 0:
                    with torch.no_grad():
                        self.validate(step=step)
                        pass

                # Get Training batch
                batch.to(device=self.device)

                # Communicate and get translation
                self.fr_en_agent.eval()
                self.en_de_agent.eval()
                trans_en, trans_en_lengths, trans_de, trans_de_lengths, trans_en_gr, trans_en_gr_lengths = \
                    self.communicate(batch)

                # Logging train communication
                if step % self.opt.logging_steps == 0:
                    self.logging_training(batch, de_train_stats, en_train_stats,
                                          step, trans_de, trans_de_lengths, trans_en_gr)

                # Training
                self.fr_en_agent.train()
                self.en_de_agent.train()

                # Get Germany Loss
                de_batch_results = process_batch_update_stats(src=trans_en[:, 1:],
                                                              src_lengths=trans_en_lengths - 1,
                                                              tgt=batch.de, tgt_lengths=batch.de_lengths,
                                                              stats_rpt=de_train_stats, agent=self.en_de_agent,
                                                              criterion=self.de_criterion)

                # Evaluate fr_en agent in training
                process_batch_update_stats(src=batch.fr, src_lengths=batch.fr_lengths,
                                           tgt=batch.en, tgt_lengths=batch.en_lengths,
                                           stats_rpt=en_train_stats, agent=self.fr_en_agent,
                                           criterion=self.en_criterion)

                # Train en_de_agent
                en_de_loss = de_batch_results[1]
                avg_en_de_loss = en_de_loss / de_batch_results[0].n_words
                self.en_de_optimizer.zero_grad()
                avg_en_de_loss.backward()
                self.en_de_optimizer.step()

                # Prepare batch for fr_en agent
                logprobs, actions, masks, values = self.evaluate_actions(batch, trans_en, trans_en_lengths)
                returns = self.get_rewards(de_batch_results, trans_en, trans_en_lengths).unsqueeze(-1)

                # Train value network
                adv = returns - values
                value_loss = self.opt.v_coeff * adv.masked_select(masks.byte()).pow(2).mean()
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                # Train with PPO
                # Prepare the mini-batch generator
                ppo_generator = self.get_ppo_generator(logprobs=logprobs,
                                                       masks=masks,
                                                       values=values,
                                                       returns=returns,
                                                       mini_bsz=self.opt.mini_bsz)

                for epoch in range(self.opt.ppo_epochs):
                    for ppo_batch in ppo_generator():
                        # Get loss
                        policy_loss, ent_loss = self.get_ppo_loss(ppo_batch=ppo_batch,
                                                                  batch=batch,
                                                                  trans_en=trans_en,
                                                                  trans_en_lengths=trans_en_lengths)
                        # Invert ent loss if reduce it
                        if self.opt.reduce_ent:
                            fr_en_loss = policy_loss - self.opt.ent_coeff * ent_loss
                        else:
                            fr_en_loss = policy_loss + self.opt.ent_coeff * ent_loss

                        # Optimize
                        self.fr_en_optimizer.zero_grad()
                        fr_en_loss.backward()
                        self.fr_en_optimizer.step()

                # Logging and Saving
                en_train_stats = self._report_training(step, en_train_stats, train_start,
                                                       prefix='train/en',
                                                       learning_rate=self.fr_en_optimizer.learning_rate)
                de_train_stats = self._report_training(step, de_train_stats, train_start,
                                                       prefix='train/de',
                                                       learning_rate=self.en_de_optimizer.learning_rate)

                # Checkpoint
                self.checkpoint(step)

                # Increment step
                step += 1
                if step > self.opt.train_steps:
                    break

    def evaluate_actions(self, batch, trans_en, trans_en_lengths):
        """ Return the action logprobs of translated english, actions, values, masks
            :param batch: An batch of data
            :param trans_en: [b, len]
            :parma trans_en_lengths: [b]
            :return logprobs: [b, len, num_action]
                    actions: [b, len]
                    masks: [b, len]
                    values: [b, len]
        """
        logprobs, actions, masks = self.fr_en_agent(src=batch.fr, src_lengths=batch.fr_lengths,
                                                    tgt=trans_en, tgt_lengths=trans_en_lengths)
        values = self.value_net(src=batch.fr, src_lengths=batch.fr_lengths,
                                tgt=trans_en, tgt_lengths=trans_en_lengths)
        return logprobs, actions, masks, values

    @staticmethod
    def get_ppo_generator(logprobs, masks, values, returns, mini_bsz):
        """ Return a generator that would generator PPO mini batch, by randomly sample `mini_bsz` sentences
            :param logprobs: [b, len]
            :param masks: [b, len]
            :param values: [b, len]
            :param returns: [b, len]
            :param mini_bsz: An int. Batch size of each mini batch
        """
        # Detach stuff
        logprobs = logprobs.detach()
        values = values.detach()
        returns = returns.detach()

        # Compute adv
        # Normalize adv with only unmasked
        advs = (returns - values)
        masked_advs = torch.masked_select(advs, mask=masks.byte())
        advs = (advs - masked_advs.mean()) / masked_advs.std()

        # get index
        total_size = logprobs.size(0)
        index = [i for i in range(total_size)]

        def generator():
            """ The actual generator """
            random.shuffle(index)
            index_tensor = torch.tensor(index, device=advs.device)
            start = 0
            while start < total_size:
                batch_index = index_tensor[start: start + mini_bsz]
                yield PPOBatch(old_logprobs=logprobs.index_select(0, batch_index),
                               returns=returns.index_select(0, batch_index),
                               advs=advs.index_select(0, batch_index),
                               batch_idx=batch_index)
                start += mini_bsz

        return generator

    def get_ppo_loss(self, ppo_batch, batch, trans_en, trans_en_lengths):
        """ Compute PPO loss for the ppo minibatch
            :param ppo_batch: An instance of PPOBatch
            :param batch: The batch containing agent inputs
            :param trans_en: [b, len]. The action/English sentence outputted
            :param trans_en_lengths: [b]. The length of English sentence.
            :return: policy_loss, entropy_loss
        """
        assert isinstance(ppo_batch, PPOBatch)
        idx = ppo_batch.batch_idx
        logprobs, actions, masks = self.fr_en_agent(src=batch.fr.index_select(0, idx),
                                                    src_lengths=batch.fr_lengths.index_select(0, idx),
                                                    tgt=trans_en.index_select(0, idx),
                                                    tgt_lengths=trans_en_lengths.index_select(0, idx))

        # Index and mask both logprobs
        # [b, len]
        masked_new_logprobs = torch.masked_select(self.slice_logprobs(logprobs, actions),
                                                  mask=masks.byte())
        masked_old_logprobs = torch.masked_select(self.slice_logprobs(ppo_batch.old_logprobs, actions),
                                                  mask=masks.byte())

        # Masking the advantage
        masked_advs = torch.masked_select(ppo_batch.advs, mask=masks.byte())

        # Compute policy loss
        ratio = torch.exp(masked_new_logprobs - masked_old_logprobs)
        surr1 = ratio * masked_advs
        surr2 = torch.clamp(ratio,
                            1. - self.opt.clip_param,
                            1. + self.opt.clip_param) * masked_advs
        policy_loss = -torch.min(surr1, surr2).mean()

        #  Get ent loss
        ent = -torch.sum(logprobs * logprobs.exp(), dim=-1)  # [bsz, seq_len]
        ent_loss = -ent.masked_select(masks.byte()).mean()
        return policy_loss, ent_loss
