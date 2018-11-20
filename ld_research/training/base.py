""" A base trainer
"""
from abc import ABC, abstractmethod
import os
import json
from tensorboardX import SummaryWriter
import argparse
import glob
import torch

from ld_research.settings import LOGGER, add_file_handler


class BaseTrainer(ABC):
    """ Base trainer Define some interface and utilities """
    def __init__(self, opt):
        """ Load and get latest opt """
        self.opt = opt

        # Load opt
        if os.path.exists(self.opt.save_dir):
            self.load_opt(self.opt.save_dir)
        else:
            os.makedirs(self.opt.save_dir)

        # Get writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.opt.save_dir, 'tensorboard'))

        # Add log file.
        add_file_handler(os.path.join(self.opt.save_dir, 'training.log'))

        # Save Opt
        self.save_opt()
        self.device = torch.device(self.opt.device)

    def save_opt(self):
        """ Save the opt to save_dir """
        with open(os.path.join(self.opt.save_dir, 'opt.json'), 'w') as f:
            f.write(json.dumps(self.opt.__dict__))

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

    def get_latest_checkpoint(self):
        """ Return the path of latest checkpoint from the save_dir. None if no checkpoint found """
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

    @abstractmethod
    def start_training(self):
        """ Main training logic """
        pass

    @abstractmethod
    def checkpoint(self, step):
        """ Save checkpoint at step """
        pass

    @abstractmethod
    def _build_data_loaders(self):
        """ Build dataloaders """
        pass

    @abstractmethod
    def _build_models(self, ckpt=None):
        """ Build Model from ckpt """
        pass

    @abstractmethod
    def _build_optimizers(self, ckpt=None):
        """ Build optimizers """
        pass
