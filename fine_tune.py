""" Pretrain the agent on IWSLT
"""
import argparse
from ld_research.settings import FR, EN, DE
from ld_research.training.iwslt import Trainer

if __name__ == '__main__':
    """ Parse from command """
    parser = argparse.ArgumentParser('fine_tune')
    parser.add_argument('-fr_en_save_dir', required=True,
                        default=None, help='path to the fr-en save_dir')
    parser.add_argument('-fr_en_ckpt', default=None,
                        help='path to fr-en ckpt. None use latest')
    parser.add_argument('-en_de_save_dir', required=True,
                        default=None, help='path to the en-de save_dir')
    parser.add_argument('-en_de_ckpt', default=None,
                        help='path to fr-en ckpt. None use latest')


    opt = parser.parse_args()

    trainer = Trainer(opt)
    trainer.start_training()
