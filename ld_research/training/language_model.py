""" Trainer for language modelling
"""
import os
import torch
import time

from ld_research.training.base import BaseTrainer
from ld_research.training.utils import NMTLoss
from ld_research.settings import LOGGER, FR, EN
from ld_research.text import IWSLTDataloader, IWSLTDataset, Vocab
from ld_research.model import LanguageModel
from ld_research.training.optimizers import Optimizer
from ld_research.training.utils import StatisticsReport


class Trainer(BaseTrainer):
    """ For language modelling """
    def __init__(self, opt):
        """ A constructor """
        super(Trainer, self).__init__(opt)
        latest_ckpt = self.get_latest_checkpoint()
        self._build_data_loaders()
        self._build_models(latest_ckpt)
        self._build_optimizers(latest_ckpt)
        self.criterion = NMTLoss(label_smoothing=0.,
                                 tgt_vocab_size=len(self.vocab),
                                 device=self.device)

    def start_training(self):
        """ Train """
        self.agent.train()
        step = self.optimizer.curr_step + 1
        LOGGER.info('Start training with step {}...'.format(step))
        train_stats = StatisticsReport()
        train_start = time.time()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):
                batch.to(device=self.device)
                batch_results = self.process_batch(en=batch.tgt,
                                                   en_lengths=batch.tgt_lengths,
                                                   stats_rpt=train_stats)
                batch_stats = batch_results[0]
                batch_loss = batch_results[1]

                # Backward and learing
                self.optimizer.zero_grad()
                (batch_loss / batch_stats.n_words).backward()
                self.optimizer.step()

                if (step + 1) % self.opt.valid_steps == 0:
                    with torch.no_grad():
                        self.validate(step=step)
                    self.agent.train()

                # Logging and saving
                train_stats = self._report_training(step, train_stats, train_start)
                self.checkpoint(step)

                # increment step
                step += 1

    def validate(self, step):
        """ Validation """
        self.agent.eval()
        valid_stats = StatisticsReport()
        for batch in self.valid_loader:
            batch.to(device=self.device)
            self.process_batch(en=batch.tgt,
                               en_lengths=batch.tgt_lengths,
                               stats_rpt=valid_stats)

        # Reporting
        LOGGER.info('Validation perplexity: %g' % valid_stats.ppl())
        LOGGER.info('Validation accuracy: %g' % valid_stats.accuracy())
        valid_stats.log_tensorboard(prefix='valid',
                                    learning_rate=self.optimizer.learning_rate,
                                    step=step,
                                    writer=self.writer)

    def _report_training(self, step, train_stats, train_start):
        """ Report the training """
        if (step + 1) % self.opt.logging_steps == 0:
            train_stats.output(step=step,
                               learning_rate=self.optimizer.learning_rate,
                               num_steps=self.opt.train_steps,
                               train_start=train_start)
            train_stats.log_tensorboard(prefix='train', writer=self.writer,
                                        learning_rate=self.optimizer.learning_rate,
                                        step=step)
            train_stats = StatisticsReport()
        return train_stats

    def checkpoint(self, step):
        """ Maybe do the checkpoint of model """
        if (step + 1) % self.opt.checkpoint_steps == 0:
            LOGGER.info('Checkpoint step {}...'.format(step))
            checkpoint = {'agent': self.agent.state_dict(),
                          'optimizer': self.optimizer}
            ckpt_name = 'checkpoint.{}.pt'.format(step)
            ckpt_path = os.path.join(self.opt.save_dir, ckpt_name)
            torch.save(checkpoint, ckpt_path)

    def _build_data_loaders(self):
        """ Get data loader """
        src_lang, tgt_lang = FR, EN

        # train valid and test
        if self.opt.debug:
            LOGGER.info('Debug mode, overfit on test data...')
            train_set = IWSLTDataset(src_lang, tgt_lang, 'test')
        else:
            train_set = IWSLTDataset(src_lang, tgt_lang, 'train')
        valid_set = IWSLTDataset(src_lang, tgt_lang, 'valid')
        self.train_loader = IWSLTDataloader(train_set, batch_size=self.opt.batch_size,
                                            shuffle=True, num_workers=1)
        self.valid_loader = IWSLTDataloader(valid_set, batch_size=self.opt.batch_size,
                                            shuffle=False, num_workers=1)
        self.vocab = Vocab(EN)

    def _build_models(self, ckpt=None):
        """ build language model """
        self.agent = LanguageModel()
        if ckpt:
            self.agent.load_state_dict(ckpt['agent'])
        else:
            self.agent.initialize(self.opt.param_init)
        self.agent.to(device=self.device)

    def _build_optimizers(self, ckpt=None):
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

    def process_batch(self, en, en_lengths, stats_rpt):
        """ Given batch process it and update stats report
            :return batch_stats, batch_loss
        """
        logprobs, targets, masks = self.agent(en, en_lengths)
        loss = self.criterion(logprobs.view(-1, logprobs.size(-1)),
                              targets.contiguous().view(-1))
        loss = loss.view(logprobs.size(0), logprobs.size(1))
        batch_loss = torch.sum(loss * masks)

        # For stats
        num_words = torch.sum(masks).item()
        num_corrects = StatisticsReport.get_num_corrects(logprobs=logprobs,
                                                         targets=targets,
                                                         masks=masks)
        batch_stats = StatisticsReport(loss=batch_loss.item(),
                                       n_words=num_words,
                                       n_correct=num_corrects,
                                       n_src_words=num_words)
        stats_rpt.update(batch_stats)
        return batch_stats, batch_loss


