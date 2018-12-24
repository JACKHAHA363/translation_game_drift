""" Run Finetuning with PPO
"""
import torch
from collections import namedtuple
import random

from ld_research.training.finetune import Trainer as PGTrainer


class PPOBatch(namedtuple('PPOBatch', ('old_logprobs',
                                       'returns',
                                       'advs',
                                       'batch_idx'))):
    """ A data structure to hold the information need to train each PPO batch """
    pass


class Trainer(PGTrainer):
    """ Use PPO to train """

    def optimize_fr_en_agent(self, batch, trans_en, trans_en_lengths, de_batch_results):
        """ Train fr_en agent with this batch. Use PPO
            :param batch: The current batch
            :param trans_en: translated english
            :param trans_en_lengths: length of translated english
            :param de_batch_results: The german results using `trans_en`
        """
        # Prepare batch for fr_en agent
        logprobs, actions, masks, values = self.evaluate_actions(batch, trans_en, trans_en_lengths)
        returns = self.get_rewards(de_batch_results, trans_en, trans_en_lengths).unsqueeze(-1)

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
        ratio = torch.exp(masked_new_logprobs - masked_new_logprobs.detach())
        surr1 = ratio * masked_advs
        surr2 = torch.clamp(ratio,
                            1. - self.opt.clip_param,
                            1. + self.opt.clip_param) * masked_advs
        policy_loss = -torch.min(surr1, surr2).mean()

        #  Get ent loss
        ent = -torch.sum(logprobs * logprobs.exp(), dim=-1)  # [bsz, seq_len]
        ent_loss = -ent.masked_select(masks.byte()).mean()
        return policy_loss, ent_loss
