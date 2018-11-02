""" Trainer objects for tinetune
"""
import os
import json
import argparse
import glob
import torch
from itertools import chain

from ld_research.settings import LOGGER, FR, EN, DE
from ld_research.text import Multi30KLoader, Multi30KDataset
from ld_research.model import Agent, ValueNetwork

class Trainer:
    """ Fine tuner """
    def __init__(self, opt):
        """ Constructor """
        self.opt = opt
        if os.path.exists(self.opt.save_dir):
            self._load_opt(self.opt.save_dir)
        else:
            os.makedirs(self.opt.save_dir)

        ckpt = self._detect_latest_ckpt()
        self.device = torch.device(self.opt.device)
        self._build_dataloaders()
        self._build_models(ckpt)

    def _process_batch(self, batch):
        """ Process a batch of data """
        batch.to(self.device)

        # Get intermediate english
        # English length contrain to be less than French
        self.fr_en_agent.batch_translate(src=batch.fr,
                                         src_lengths=batch.fr_lengths,
                                         max_lengths=batch.fr_lengths)

    def _load_opt(self, save_dir):
        """ load save_dir """
        LOGGER.info('Loading option in {}...'.format(save_dir))
        opt_json = os.path.join(save_dir, 'opt.json')
        if os.path.exists(opt_json):
            LOGGER.info('Found json file in {}...'.format(save_dir))
            with open(opt_json, 'r') as f:
                opt_dict = json.load(f)
                self.opt = argparse.Namespace(**opt_dict)
        else:
            LOGGER.info('No json file in {}. Use current opt...'.format(save_dir))

    def _detect_latest_ckpt(self):
        """ Detect latest ckpt """
        save_dir = self.opt.save_dir
        all_ckpt = glob.glob(os.path.join(save_dir, 'checkpoint.*.pt'))
        latest_ckpt = None
        if all_ckpt:
            all_steps = [int(os.path.basename(ckpt_name).split('.')[1]) for ckpt_name in all_ckpt]
            latest_step = max(all_steps)
            latest_ckpt_path = os.path.join(save_dir, 'checkpoint.{}.pt'.format(latest_step))
            latest_ckpt = torch.load(latest_ckpt_path,
                                     map_location=lambda storage, loc: storage)
        return latest_ckpt

    def _build_dataloaders(self):
        """ Build datasets """

        if self.opt.debug:
            LOGGER.info('Debug mode, overfit on test data...')
            train_set = Multi30KDataset('test')
        else:
            train_set = Multi30KDataset('train')
        valid_set = Multi30KDataset('valid')
        self.train_loader = Multi30KLoader(train_set, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=1)
        self.valid_loader = Multi30KLoader(valid_set, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=1)
        self.vocabs = valid_set.vocabs

    def _build_models(self, ckpt=None):
        """ Build model """
        self.value_net = ValueNetwork(src_vocab=self.vocabs[FR],
                                      tgt_vocab=self.vocabs[EN],
                                      opt=self.opt)
        self.fr_en_agent = Agent(src_vocab=self.vocabs[FR],
                                 tgt_vocab=self.vocabs[EN],
                                 opt=self.opt)
        self.en_de_agent = Agent(src_vocab=self.vocabs[EN],
                                 tgt_vocab=self.vocabs[DE],
                                 opt=self.opt)
        if ckpt:
            self.value_net.load_state_dict(ckpt['value_net'])
            self.fr_en_agent.load_state_dict(ckpt['fr_en_agent'])
            self.en_de_agent.load_state_dict(ckpt['en_de_agent'])
        else:
            for p in chain(self.value_net.parameters(),
                           self.fr_en_agent.parameters(),
                           self.en_de_agent.parameters()):
                p.data.uniform_(-self.opt.param_init, self.opt.param_init)

        # Move to gpu/cpu
        self.value_net.to(device=self.device)
        self.fr_en_agent.to(device=self.device)
        self.en_de_agent.to(device=self.device)
