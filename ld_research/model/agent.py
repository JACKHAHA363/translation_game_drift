""" The general interface of an model
"""
import torch
from torch.nn import functional as F

from ld_research.model.grus import GRUDecoder, GRUEncoder
from ld_research.text import Vocab
from ld_research.model.utils import sequence_mask
from ld_research.settings import BOS_WORD, EOS_WORD

class Agent(torch.nn.Module):
    """ The model. A seq to seq model """
    def __init__(self, src_vocab, tgt_vocab, opt):
        """ Constructor
            :param src_vocab: source language_vocab. An instance of `Vocab`
            :param tgt_vocab: target language_vocab. An instance of `Vocab`
            :param opt: The hyparparameters
        """
        super(Agent, self).__init__()
        assert isinstance(src_vocab, Vocab)
        assert isinstance(tgt_vocab, Vocab)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_module('src_emb', torch.nn.Embedding(len(src_vocab),
                                                      opt.emb_size))
        self.add_module('tgt_emb', torch.nn.Embedding(len(tgt_vocab),
                                                      opt.emb_size))
        self.add_module('encoder', GRUEncoder(self.src_emb, hidden_size=opt.hidden_size))
        self.add_module('decoder', GRUDecoder(self.tgt_emb, hidden_size=opt.hidden_size))

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None, with_align=False):
        """ Forwarding
            :param src [bsz, src_len]
            :param tgt [bsz, tgt_len]
            :param src_lengths [bsz]
            :param tgt_lengths [bsz]
            :param with_align: Whether or not output alignments
            :return: logprobs, targets, mask, alignment (depends)
                logprobs: The logits result. [bsz, tgt_len-1, tgt_vocab]
                targets: The targets to match. [bsz, tgt_len-1]
                mask: The sequence mask. [bsz, tgt_len-1]. None if tgt_lengths is None
                alignments: The attention alignments [bsz, tgt_len-1, src_len]
        """
        states, memory = self.encoder.encode(src)

        # Set up input and output for decoder
        tgt_in = tgt[:, :-1]
        logits, alignments = self.decoder.teacher_forcing(targets=tgt_in,
                                                          memory=memory,
                                                          states=states,
                                                          memory_lengths=src_lengths)
        # Get outputs
        tgt_out = tgt[:, 1:]
        masks = None if tgt_lengths is None else \
            sequence_mask(lengths=tgt_lengths - 1,
                          max_len=logits.size(1))
        masks = masks.float().to(device=self.device)
        logprobs = F.log_softmax(logits, dim=-1)
        outputs = [logprobs, tgt_out, masks]
        if with_align:
            outputs += [alignments]
        return tuple(outputs)

    @property
    def device(self):
        """ Return the device """
        first_param = next(self.parameters())
        return first_param.device

    def batch_translate(self, src, src_lengths, max_len=None):
        """ Batch of sentences. Already padded and turn into tensor """
        bos_id = self.tgt_vocab.get_index(BOS_WORD)
        eos_id = self.tgt_vocab.get_index(EOS_WORD)
        states, memory = self.encoder.encode(src)
        sample_ids, sample_lengths = self.decoder.greedy_decoding(bos_id=bos_id,
                                                                  eos_id=eos_id,
                                                                  memory=memory,
                                                                  memory_lengths=src_lengths,
                                                                  states=states,
                                                                  max_steps=max_len)
        return sample_ids, sample_lengths

    def translate(self, src, max_len=None):
        """ translate a single sentence.
            :param src: A tensor. [len]
            :param max_len: Maximum sentence length. If None, no limit.
            :return (tgt, length). A tensor of shape ([len], [])
        """
        bos_id = self.tgt_vocab.get_index(BOS_WORD)
        eos_id = self.tgt_vocab.get_index(EOS_WORD)

        states, memory = self.encoder.encode(src.unsqueeze(0))
        sample_ids, sample_lengths = self.decoder.greedy_decoding(bos_id=bos_id,
                                                                  eos_id=eos_id,
                                                                  memory=memory,
                                                                  states=states,
                                                                  max_steps=max_len)
        return sample_ids.squeeze(0), sample_lengths.squeeze(0)
