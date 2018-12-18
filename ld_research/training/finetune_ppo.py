""" Run Finetuning with PPO
"""
import torch
import time

from ld_research.training.utils import StatisticsReport, process_batch_update_stats
from ld_research.training.finetune import Trainer as PGTrainer


class PPOCommuTrainer(PGTrainer):
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

                # Get log-probs
                action_logprobs, actions, action_masks = self.fr_en_agent(src=batch.fr, src_lengths=batch.fr_lengths,
                                                                          tgt=trans_en, tgt_lengths=trans_en_lengths)
                values = self.value_net(src=batch.fr, src_lengths=batch.fr_lengths,
                                        tgt=trans_en, tgt_lengths=trans_en_lengths)

                # Train fr_en_agent
                # [bsz, seq_len]
                rewards = self.get_rewards(de_batch_results, trans_en, trans_en_lengths)

                # reinforce
                # [bsz, seq_len]
                adv = rewards.unsqueeze(-1) - values
                pg_loss, value_loss, ent_loss = self.get_rl_loss(adv=adv,
                                                                 logprobs=action_logprobs,
                                                                 actions=actions,
                                                                 action_masks=action_masks)

                # Invert ent loss if reduce it
                if self.opt.reduce_ent:
                    fr_en_loss = pg_loss + self.opt.v_coeff * value_loss - self.opt.ent_coeff * ent_loss
                else:
                    fr_en_loss = pg_loss + self.opt.v_coeff * value_loss + self.opt.ent_coeff * ent_loss
                self.value_optimizer.zero_grad()
                self.fr_en_optimizer.zero_grad()
                fr_en_loss.backward()
                self.value_optimizer.step()
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

        @staticmethod
        def masked_action_logprobs(batch, trans_en, trans_en_lengths):
            """ Return the masked action logprobs of translated english
                :param batch: An batch of data
                :param trans_en: [b, len]
                :parma trans_en_lengths: [b]
                :return action_logprobs: [b, len]
            """
            action_logprobs, actions, action_masks = self.fr_en_agent(src=batch.fr, src_lengths=batch.fr_lengths,
                                                                      tgt=trans_en, tgt_lengths=trans_en_lengths)

            # index get policy log probs
            # [bsz, seq_len]
            logprobs_flat = action_logprobs.view(-1, action_logprobs.size(2))
            actions_flat = actions.contiguous().view(-1, 1)
            indexed_logprobs = logprobs_flat.gather(1, actions_flat).squeeze(-1)
            policy_logprobs = indexed_logprobs.view(adv.size(0), adv.size(1))
            return torch.masked_select(policy_logprobs, mask=action_masks.byte())
