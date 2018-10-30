""" The general interface of an model
"""
import torch
from torch.nn import functional as F
from ld_research.model.grus import GRUDecoder, GRUEncoder
from ld_research.text import Vocab
from ld_research.model.utils import sequence_mask

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
        self.criterion = NMTLoss(label_smoothing=opt.label_smoothing,
                                 tgt_vocab_size=len(tgt_vocab))

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

    def compute_loss(self, src, tgt, src_lengths=None, tgt_lengths=None):
        """ Compute the mean loss """
        logprobs, targets, masks = self.forward(src, tgt, src_lengths, tgt_lengths)
        loss = self.criterion(logprobs.view(-1, len(self.tgt_vocab)),
                              targets.contiguous().view(-1))
        loss = loss.view(logprobs.size(0), logprobs.size(1))
        if masks is not None:
            nb_non_masks = torch.sum(masks == 0).float()
            loss_avg = torch.sum(loss * masks) / nb_non_masks
        else:
            loss_avg = torch.mean(loss)
        return loss_avg

    @property
    def device(self):
        """ Return the device """
        first_param = next(self.parameters())
        return first_param.device

class NMTLoss(torch.nn.Module):
    """ With label smoothing,
        KL-divergence between q_{smoothed ground truth prob.}(w)
        and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, device=None):
        """
            :param label_smoothing: The degree of label smoothing
            :param tgt_vocab_size: size of target vocabulary
            :param device: The device of this loss
        """
        assert 0.0 < label_smoothing <= 1.0
        super(NMTLoss, self).__init__()
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot = one_hot.to(device=device)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
            :param output (FloatTensor): batch_size x n_classes
            :param target (LongTensor): batch_size
            :param loss: (bsz)
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.sum(F.kl_div(output, model_prob, reduction='none'), -1)
