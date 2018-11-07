""" The trainer object for pretrain on iwslt
"""
import os
import torch
import time

from ld_research.settings import LOGGER
from ld_research.text import IWSLTDataset, IWSLTDataloader
from ld_research.model import Agent
from ld_research.training.optimizers import Optimizer
from ld_research.training.base import BaseTrainer
from ld_research.training.utils import process_batch_update_stats, NMTLoss, StatisticsReport

class Trainer(BaseTrainer):
    """ An object for doing pretraining """
    def __init__(self, opt):
        """ A constructor """
        super(Trainer, self).__init__(opt)

        # Checkpoint
        latest_ckpt = self.get_latest_checkpoint()

        # Get data/model
        self._build_data_loaders()
        self._build_models(latest_ckpt)

        # Get optimizer
        self._build_optimizers(latest_ckpt)

        # Get Loss
        self.criterion = NMTLoss(label_smoothing=self.opt.label_smoothing,
                                 tgt_vocab_size=len(self.train_loader.tgt_vocab),
                                 device=self.device)

    def start_training(self):
        """ Start training """
        step = self.optimizer.curr_step + 1
        LOGGER.info('Start training with step {}...'.format(step))
        train_stats = StatisticsReport()
        train_start = time.time()
        while step <= self.opt.train_steps:
            for i, batch in enumerate(self.train_loader):
                batch.to(device=self.device)
                batch_results = process_batch_update_stats(src=batch.src,
                                                           tgt=batch.tgt,
                                                           src_lengths=batch.src_lengths,
                                                           tgt_lengths=batch.tgt_lengths,
                                                           stats_rpt=train_stats,
                                                           agent=self.agent,
                                                           criterion=self.criterion)
                batch_stats = batch_results[0]
                total_loss = batch_results[1]

                # Backward and learing
                self.optimizer.zero_grad()
                (total_loss / batch_stats.n_words).backward()
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
        """ Validatation """
        self.agent.eval()
        valid_stats = StatisticsReport()
        actuals = []
        references = []
        for batch in self.valid_loader:
            batch.to(device=self.device)
            process_batch_update_stats(src=batch.src,
                                       tgt=batch.tgt,
                                       src_lengths=batch.src_lengths,
                                       tgt_lengths=batch.tgt_lengths,
                                       stats_rpt=valid_stats,
                                       agent=self.agent,
                                       criterion=self.criterion)
            preds, pred_lengths = self.agent.batch_translate(batch.src,
                                                             batch.src_lengths,
                                                             max_lengths=100,
                                                             method=self.opt.sample_method)
            actuals += self.tgt_vocab.to_sentences(preds.tolist())
            references += [[tgt_sent] for tgt_sent in self.tgt_vocab.to_sentences(batch.tgt.tolist())]

        # Reporting
        LOGGER.info('Validation perplexity: %g' % valid_stats.ppl())
        LOGGER.info('Validation accuracy: %g' % valid_stats.accuracy())
        valid_stats.log_tensorboard(prefix='valid',
                                    learning_rate=self.optimizer.learning_rate,
                                    step=step,
                                    writer=self.writer)
        LOGGER.info('Compute BLEU Score...')
        valid_stats.report_bleu_score(references, actuals, self.writer,
                                      prefix='valid', step=step)

    def _build_data_loaders(self):
        """ Build data loaders """
        src_lang, tgt_lang = self.opt.src_lang, self.opt.tgt_lang

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
        self.src_vocab = self.train_loader.src_vocab
        self.tgt_vocab = self.train_loader.tgt_vocab

    def _build_models(self, ckpt=None):
        """ Build models """
        self.agent = Agent(src_vocab=self.src_vocab,
                           tgt_vocab=self.tgt_vocab,
                           opt=self.opt)
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
