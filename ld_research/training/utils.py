""" Utilities for training
"""
import time
import torch
from torch.nn import functional as F
import math
from nltk.translate.bleu_score import corpus_bleu

from ld_research.settings import LOGGER


class StatisticsReport(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, n_src_words=0):
        """ Constructor """
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = n_src_words
        self.start_time = time.time()

    @staticmethod
    def get_batch_stats(tgt_lengths, src_lengths, logprobs, loss, masks, targets):
        """ Return a batch statistics """
        num_words = torch.sum(tgt_lengths - 1).item()
        num_corrects = StatisticsReport.get_num_corrects(logprobs=logprobs,
                                                         targets=targets,
                                                         masks=masks)
        num_src_words = torch.sum(src_lengths).item()
        batch_stats = StatisticsReport(loss=loss.item(),
                                       n_words=num_words,
                                       n_correct=num_corrects,
                                       n_src_words=num_src_words)
        return batch_stats

    @staticmethod
    def get_num_corrects(logprobs, targets, masks):
        """ Get the number of corrects
            :param logprobs: [bsz, seq_len, vocab_size]
            :param targets: [bsz, seq_len]
            :param masks: [bsz, seq_len] of 0s and 1s
            :return: The number correct
        """
        _, preds = torch.max(logprobs, dim=-1)
        corrects = (preds == targets).float() * masks
        return torch.sum(corrects).item()

    def update(self, stat):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, train_start, prefix=''):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           num_steps (int): total batches
           learning_rate (float): learning rate
           train_start: The starting time of training
           prefix: The title
        """
        t = self.elapsed_time()
        LOGGER.info(
            ("%s Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (prefix,
               step, num_steps,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - train_start))

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)

    @staticmethod
    def report_bleu_score(references, hypotheses, writer, prefix, step):
        """ Report bleu scores """
        bleu_weights = {'bleu_1': (1.0, 0, 0, 0),
                        'bleu_2': (0.5, 0.5, 0, 0),
                        'bleu_3': (1/3., 1/3., 1/3., 0),
                        'bleu_4': (0.25, 0.25, 0.25, 0.25)}
        scores = {name: corpus_bleu(list_of_references=references,
                                    hypotheses=hypotheses, weights=weights) * 100
                  for name, weights in bleu_weights.items()}
        for name, score in scores.items():
            LOGGER.info('[{}] {}: {:.2f}'.format(prefix, name, score))
            writer.add_scalar('{}/{}'.format(prefix, name), score, step)


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
        super(NMTLoss, self).__init__()
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot = one_hot.to(device=device)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """ compute the loss
            :param output (FloatTensor): batch_size x n_classes
            :param target (LongTensor): batch_size
            :param loss: (bsz)
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.sum(F.kl_div(output, model_prob, reduction='none'), -1)


def process_batch_update_stats(src, src_lengths, tgt, tgt_lengths, stats_rpt, agent, criterion):
    """ Process the batch and update the statistics `stats_rpt`
        :param src: Tensor [bsz, src_len]
        :param src_lengths: Tensor [bsz]
        :param tgt: Tensor [bsz, src_len]
        :param tgt_lengths: Tensor [bsz]
        :param stats_rpt: An instance of StatisticsReport
        :param agent: An instance of `Agent`
        :param criterion: An instance of `NMTLoss`
        return (batch_stats, loss, logprobs, targets, masks)
            batch_stats: An instance of `StatisticsReport`
            total_loss: A scalar
            loss: [bsz, seq_len]
            logprobs: [bsz, seq_len, vocab_size]
            targets, masks: [bsz, seq_len]
    """
    logprobs, targets, masks = agent(src=src,
                                     tgt=tgt,
                                     src_lengths=src_lengths,
                                     tgt_lengths=tgt_lengths)
    loss = criterion(logprobs.view(-1, logprobs.size(-1)),
                     targets.contiguous().view(-1))
    loss = loss.view(logprobs.size(0), logprobs.size(1))
    if masks is not None:
        total_loss = torch.sum(loss * masks)
    else:
        total_loss = torch.sum(loss)
    batch_stats = StatisticsReport.get_batch_stats(tgt_lengths=tgt_lengths,
                                                   src_lengths=src_lengths,
                                                   logprobs=logprobs,
                                                   loss=total_loss,
                                                   masks=masks,
                                                   targets=targets)
    stats_rpt.update(batch_stats)
    return batch_stats, total_loss, loss, logprobs, targets, masks
