""" The trainer object for pretrain on iwslt
"""
import os
import torch
import argparse
import json
import time
import math
from tensorboardX import SummaryWriter
import glob
from torch.nn import functional as F
from nltk.translate.bleu_score import corpus_bleu

from ld_research.settings import FR, EN, DE, LOGGER, add_file_handler
from ld_research.text import IWSLTDataloader, IWSLTDataset
from ld_research.model import Agent
from ld_research.training.optimizers import Optimizer

class Trainer:
    """ An object for doing pretraining """
    def __init__(self, opt):
        """ A constructor """
        self.opt = opt
        assert (opt.src_lang, opt.tgt_lang) in [(FR, EN), (EN, DE)]

        # Load opt
        if os.path.exists(self.opt.save_dir):
            self.load_opt(self.opt.save_dir)
        else:
            os.makedirs(self.opt.save_dir)

        # Checkpoint
        latest_ckpt_path = self._get_latest_checkpoint_path()
        latest_ckpt = None
        if latest_ckpt_path:
            latest_ckpt = torch.load(latest_ckpt_path,
                                     map_location=lambda storage, loc: storage)

        # Get data/model
        self._build_dataloader_and_model(latest_ckpt)

        # Get optimizer
        self._build_optimizer(latest_ckpt)

        # Get Loss
        self.criterion = NMTLoss(label_smoothing=self.opt.label_smoothing,
                                 tgt_vocab_size=len(self.train_loader.tgt_vocab),
                                 device=self.agent.device)

        # Get writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.opt.save_dir, 'tensorboard'))

        # Save Opt
        self.save_opt()

        # Add log file.
        add_file_handler(os.path.join(self.opt.save_dir, 'training.log'))

    def start_training(self):
        """ Start training """
        step = self.optimizer.curr_step + 1
        LOGGER.info('Start training with step {}...'.format(step))
        train_stats = StatisticsReport()
        train_start = time.time()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):
                batch.to(device=self.device)
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

                # Backward and learing
                self.optimizer.zero_grad()
                (loss / batch_stats.n_words).backward()
                self.optimizer.step()

                if (step + 1) % self.opt.valid_steps == 0:
                    with torch.no_grad():
                        self.validate(step=step,
                                      train_start=train_start)
                        self.compute_bleu(step=step,
                                          train_start=train_start)
                    self.agent.train()

                # Logging and saving
                train_stats = self._report_training(step, train_stats, train_start)
                self._checkpoint(step)

                # increment step
                step += 1

    def validate(self, step, train_start):
        """ Validatation """
        self.agent.eval()
        valid_stats = StatisticsReport()

        for batch in self.valid_loader:
            batch.to(device=self.device)
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
            valid_stats.update(batch_stats)

        # Reporting
        LOGGER.info('Validation perplexity: %g' % valid_stats.ppl())
        LOGGER.info('Validation accuracy: %g' % valid_stats.accuracy())
        valid_stats.log_tensorboard(prefix='valid',
                                    learning_rate=self.optimizer.learning_rate,
                                    step=step,
                                    train_start=train_start,
                                    writer=self.writer)

    def compute_bleu(self, step, train_start):
        """ Greedy decoding """
        LOGGER.info('Compute bleu score...')
        self.agent.eval()
        actual = []
        references = []
        for batch in self.valid_loader:
            # Adding targets
            references += [[sent] for sent in self.tgt_vocab.to_sentences(ids=batch.tgt.tolist())]

            # Add predictions
            batch.to(device=self.device)
            sample_ids = self.agent.batch_translate(src=batch.src,
                                                    src_lengths=batch.src_lengths,
                                                    max_len=200)
            actual += self.tgt_vocab.to_sentences(ids=sample_ids.tolist())

        # Modify reference to have an extra list wrapper
        bleu_2 = corpus_bleu(references, actual, weights=(0.5, 0.5, 0, 0))
        bleu_4 = corpus_bleu(references, actual, weights=(0.25, 0.25, 0.25, 0.25))
        LOGGER.info('Validation bleu_2: {}'.format(bleu_2))
        LOGGER.info('Validation bleu_4: {}'.format(bleu_4))
        self.writer.add_scalar('valid/bleu2', bleu_2, step, walltime=train_start-time.time())
        self.writer.add_scalar('valid/bleu4', bleu_4, step, walltime=train_start-time.time())


    def _build_optimizer(self, ckpt=None):
        """ Get optimizer """
        saved_state_dict = None
        if ckpt:
            LOGGER.info('Loading the optimizer info from {}...'.format(self.opt.save_dir))
            self.optimizer = ckpt['optimizer']
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
                warmup_steps=opt.warmup_steps)

        # Set parameters by initialize new torch optim inside
        self.optimizer.set_parameters(params=self.agent.named_parameters())

        # Set the states
        if saved_state_dict is not None:
            self.optimizer.load_state_dict(saved_state_dict)

    def _build_dataloader_and_model(self, ckpt=None):
        """ Build data and model """
        self.device = torch.device(self.opt.device)
        src_lang, tgt_lang = self.opt.src_lang, self.opt.tgt_lang

        # train valid and test
        if self.opt.debug:
            LOGGER.info('Debug mode, overfit on validation...')
            train_set = IWSLTDataset(src_lang, tgt_lang, 'valid')
            valid_set = IWSLTDataset(src_lang, tgt_lang, 'valid')
        else:
            train_set = IWSLTDataset(src_lang, tgt_lang, 'train')
            valid_set = IWSLTDataset(src_lang, tgt_lang, 'valid')
        self.train_loader = IWSLTDataloader(train_set, batch_size=self.opt.batch_size,
                                            shuffle=True, num_workers=1)
        self.valid_loader = IWSLTDataloader(valid_set, batch_size=self.opt.batch_size,
                                            shuffle=False, num_workers=1)
        self.src_vocab = self.train_loader.src_vocab
        self.tgt_vocab = self.train_loader.tgt_vocab
        self.agent = Agent(src_vocab=self.src_vocab,
                           tgt_vocab=self.tgt_vocab,
                           opt=self.opt)
        if ckpt:
            self.agent.load_state_dict(ckpt['agent'])
        self.agent.to(device=self.device)

    def load_opt(self, save_dir):
        """ Load opt from save dir """
        LOGGER.info('Loading option in {}...'.format(save_dir))
        opt_json = os.path.join(save_dir, 'opt.json')
        if os.path.exists(opt_json):
            LOGGER.info('Found json file in {}...'.format(save_dir))
            with open(opt_json, 'r') as f:
                opt_dict = json.load(f)
                self.opt = argparse.Namespace(**opt_dict)
        else:
            LOGGER.info('No json file in {}. Use current opt...'.format(save_dir))

    def save_opt(self):
        """ Save the opt to save_dir """
        with open(os.path.join(self.opt.save_dir, 'opt.json'), 'w') as f:
            f.write(json.dumps(self.opt.__dict__))

    def _report_training(self, step, train_stats, train_start):
        """ Report the training """
        if (step + 1) % self.opt.logging_steps == 0:
            train_stats.output(step=step,
                               learning_rate=self.optimizer.learning_rate,
                               num_steps=self.opt.train_steps,
                               train_start=train_start)
            train_stats.log_tensorboard(prefix='train', writer=self.writer,
                                        learning_rate=self.optimizer.learning_rate,
                                        step=step, train_start=train_start)
            train_stats = StatisticsReport()
        return train_stats

    def _checkpoint(self, step):
        """ Maybe do the checkpoint of model """
        if (step + 1) % self.opt.checkpoint_steps == 0:
            LOGGER.info('Checkpoint step {}...'.format(step))
            checkpoint = {'agent': self.agent.state_dict(),
                          'optimizer': self.optimizer}
            ckpt_name = 'checkpoint.{}.pt'.format(step)
            ckpt_path = os.path.join(self.opt.save_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

    def _get_latest_checkpoint_path(self):
        """ Return the path of latest checkpoint from the save_dir. None if no checkpoint found """
        save_dir = self.opt.save_dir
        all_ckpt = glob.glob(os.path.join(save_dir, 'checkpoint.*.pt'))
        latest_ckpt_path = None
        if all_ckpt:
            all_steps = [int(os.path.basename(ckpt_name).split('.')[1]) for ckpt_name in all_ckpt]
            latest_step = max(all_steps)
            latest_ckpt_path = os.path.join(save_dir, 'checkpoint.{}.pt'.format(latest_step))
        return latest_ckpt_path

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
