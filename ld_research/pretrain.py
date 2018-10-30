""" Pretrain the agent on IWSLT
"""
import argparse
import os
import json
import torch
from ld_research.settings import FR, EN, DE, LOGGER
from ld_research.text.datasets import IWSLTDataset, IWSLTDataloader
from ld_research.model import Agent

class Trainer:
    """ A trainer object that coordinate model, dataset and compute stats """
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

    def train_for_one_epoch(self):
        """ Start training """
        self.agent.train()

    def _build_optimizer(self, from_ckpt=False):
        """ Get optimizer """

    def _build_dataloader_and_model(self, from_ckpt=False):
        """ Build data and model """
        src_lang, tgt_lang = self.opt.src_lang, self.opt.tgt_lang
        train_set = IWSLTDataset(src_lang, tgt_lang, 'train', device=self.opt.device)
        valid_set = IWSLTDataset(src_lang, tgt_lang, 'valid', device=self.opt.device)
        self.train_loader = IWSLTDataloader(train_set, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = IWSLTDataloader(valid_set, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=4, pin_memory=True)
        self.src_vocab = self.train_loader.src_vocab
        self.tgt_vocab = self.train_loader.tgt_vocab
        self.agent = Agent(src_vocab=self.src_vocab,
                           tgt_vocab=self.tgt_vocab,
                           opt=self.opt)
        if from_ckpt:
            self.load_model()
        self.agent.to(device=torch.device(opt.device))

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

if __name__ == '__main__':
    """ Parse from command """
    parser = argparse.ArgumentParser('Pretrain')
    parser.add_argument('-src_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    parser.add_argument('-tgt_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-label_smoothing', default=0.1, type=float)
    parser.add_argument('-device', default='cpu', type=str,
                        help='The device. None, cpu, cuda, cuda/cpu:rank')

    # optimizer
    parser.add_argument('-optim', default='sgd',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) steps have gone past
                        start_decay_steps""")
    parser.add_argument('-start_decay_steps', type=int, default=50000,
                        help="""Start decaying every decay_steps after
                        start_decay_steps""")
    parser.add_argument('-decay_steps', type=int, default=10000,
                        help="""Decay every decay_steps""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    # Saving and Logging
    parser.add_argument('--save_dir', default='./save_dir', type=str)

    opt = parser.parse_args()
    trainer = Trainer(opt)
