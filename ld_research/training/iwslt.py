""" The trainer object for pretrain on iwslt
"""
import os
import torch
import argparse
import json
import time
import math
from tensorboardX import SummaryWriter

from torch.nn import functional as F

from ld_research.settings import FR, EN, DE, LOGGER
from ld_research.text import IWSLTDataloader, IWSLTDataset
from ld_research.model import Agent
from ld_research.training.optimizers import Optimizer

class Trainer:
    """ An object for doing pretraining """
    def __init__(self, opt):
        """ A constructor """
        self.opt = opt
        assert (opt.src_lang, opt.tgt_lang) in [(FR, EN), (EN, DE)]
        from_ckpt = False
        if not os.path.exists(opt.save_dir):
            from_ckpt = True
            os.makedirs(opt.save_dir)

        # Load opt from ckpt
        if from_ckpt:
            self.load_opt(opt.save_dir)

        # Get data/model
        self._build_dataloader_and_model(from_ckpt)

        # Get optimizer
        self._build_optimizer(from_ckpt)

        # Get Loss
        self.criterion = NMTLoss(label_smoothing=self.opt.label_smoothing,
                                 tgt_vocab_size=len(self.train_loader.tgt_vocab),
                                 device=self.agent.device)

        # Get writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.opt.save_dir, 'tensorboard'))

    def start_training(self):
        """ Start training """
        LOGGER.info('Start training...')
        step = self.optimizer.curr_step + 1
        train_stats = StatisticsReport()
        train_start = time.start()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):
                logprobs, targets, masks = self.agent(src=batch.src,
                                                      tgt=batch.tgt,
                                                      src_lengths=batch.src_lengths,
                                                      tgt_lengths=batch.tgt_lengths)
                loss = self.criterion(logprobs.view(-1, len(self.tgt_vocab)),
                                      targets.contiguous().view(-1))
                loss = loss.view(logprobs.size(0), logprobs.size(1))
                if masks is not None:
                    loss = torch.sum(loss * masks)
                else:
                    loss = torch.sum(loss)
                batch_stats = StatisticsReport.get_batch_stats(batch=batch,
                                                               logprobs=logprobs,
                                                               loss=loss,
                                                               masks=masks,
                                                               targets=targets)
                train_stats.update(batch_stats)
                train_stats = self._report_training(step, train_stats)

                # Backward and learing
                self.optimizer.zero_grad()
                (loss / batch_stats.n_words).backward()
                self.optimizer.step()

                if step % self.opt.valid_steps == 0:
                    self.validate()
                step += 1

    def validate(self):
        """ Validatation """
        pass

    def _build_optimizer(self, from_ckpt=False):
        """ Get optimizer """
        saved_state_dict = None
        if from_ckpt:
            LOGGER.info('Loading the optimizer info from {}...'.format(self.opt.save_dir))
            self.optimizer = torch.load(os.path.join(self.opt.save_dir, 'latest.optimizer.pt'),
                                        map_location=lambda storage, loc: storage)
            saved_state_dict = self.optimizer.state_dict()
        else:
            opt = self.opt
            self.optimizer = Optimizer(
                opt.optim, opt.learning_rate, opt.max_grad_norm,
                lr_decay=opt.learning_rate_decay,
                start_decay_steps=opt.start_decay_steps,
                decay_steps=opt.decay_steps,
                beta1=opt.adam_beta1,
                beta2=opt.adam_beta2,
                adagrad_accum=opt.adagrad_accumulator_init,
                decay_method=opt.decay_method,
                warmup_steps=opt.warmup_steps,
                model_size=opt.rnn_size)

        # Set parameters by initialize new torch optim inside
        self.optimizer.set_parameters(params=self.agent.parameters())

        # Set the states
        if saved_state_dict is not None:
            self.optimizer.load_state_dict(saved_state_dict)

    def _build_dataloader_and_model(self, from_ckpt=False):
        """ Build data and model """
        src_lang, tgt_lang = self.opt.src_lang, self.opt.tgt_lang
        train_set = IWSLTDataset(src_lang, tgt_lang, 'train', device=self.opt.device)
        valid_set = IWSLTDataset(src_lang, tgt_lang, 'valid', device=self.opt.device)
        self.train_loader = IWSLTDataloader(train_set, batch_size=self.opt.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = IWSLTDataloader(valid_set, batch_size=self.opt.batch_size,
                                            shuffle=False, num_workers=4, pin_memory=True)
        self.src_vocab = self.train_loader.src_vocab
        self.tgt_vocab = self.train_loader.tgt_vocab
        self.agent = Agent(src_vocab=self.src_vocab,
                           tgt_vocab=self.tgt_vocab,
                           opt=self.opt)
        if from_ckpt:
            self.load_model()
        self.agent.to(device=torch.device(self.opt.device))

    def load_opt(self, save_dir):
        """ Load opt from save dir """
        with open(os.path.join(save_dir, 'opt.json'), 'r') as f:
            opt_dict = json.load(f)
            self.opt = argparse.Namespace(**opt_dict)

    def save_opt(self):
        """ Save the opt to save_dir """
        with open(os.path.join(self.opt.save_dir, 'opt.json'), 'w') as f:
            f.write(json.dumps(self.opt.__dict__))

    def load_model(self):
        """ Load from agent """
        LOGGER.info('Loading latest model from {}...'.format(self.opt.save_dir))
        state_dict = torch.load(os.path.join(self.opt.save_dir, 'latest.pt'),
                                map_location=lambda storage, loc: storage)
        self.agent.load_state_dict(state_dict)

    def _report_training(self, step, train_stats, train_start):
        """ Report the training """
        # Report every 100
        if step % 100 == 0:
            train_stats.output(step=step,
                               learning_rate=self.optimizer.learning_rate,
                               num_steps=self.opt.train_steps,
                               train_start=train_start)
            train_stats.log_tensorboard(prefix='train', writer=self.writer,
                                        learning_rate=self.optimizer.learning_rate,
                                        step=step, train_start=train_start)
            train_stats = StatisticsReport()
        return train_stats

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
    def get_batch_stats(batch, logprobs, loss, masks, targets):
        """ Return a batch statistics """
        num_words = torch.sum(batch.tgt_lengths - 1).item()
        num_corrects = StatisticsReport.get_num_corrects(logprobs=logprobs,
                                                         targets=targets,
                                                         masks=masks)
        num_src_words = torch.sum(batch.src_lengths).item()
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
        corrects = (preds == targets) * masks
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

    def output(self, step, num_steps, learning_rate, train_start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           num_steps (int): total batches
           learning_rate (float): learning rate
           train_start: The starting time of training
        """
        t = self.elapsed_time()
        LOGGER.info(
            ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - train_start))

    def log_tensorboard(self, prefix, writer, learning_rate, step, train_start):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        walltime = time.time() - train_start
        writer.add_scalar(prefix + "/xent", self.xent(), step, walltime=walltime)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step, walltime=walltime)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step, walltime=walltime)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step, walltime=walltime)
        writer.add_scalar(prefix + "/lr", learning_rate, step, walltime=walltime)

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