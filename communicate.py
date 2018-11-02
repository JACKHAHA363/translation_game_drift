""" Pretrain the agent on IWSLT
"""
import argparse
from ld_research.settings import FR, EN, DE
from ld_research.training.iwslt import Trainer

if __name__ == '__main__':
    """ Parse from command """
    parser = argparse.ArgumentParser('fine_tune')
    parser.add_argument('-save_dir', required=True, default=None,
                        help='Final output')
    parser.add_argument('-fr_en_save_dir', required=None,
                        default=None, help='path to the fr-en save_dir')
    parser.add_argument('-en_de_save_dir', required=None,
                        default=None, help='path to the en-de save_dir')
    parser.add_argument('-fr_en_ckpt', default=None,
                        help='path to fr-en ckpt. None use latest')
    parser.add_argument('-en_de_ckpt', default=None,
                        help='path to fr-en ckpt. None use latest')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-batch_size', default=64, type=int)

    # Model
    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-value_emb_size', default=256, type=int)
    parser.add_argument('-value_hidden_size', default=256, type=int)
    parser.add_argument('-param_init', default=0.1, type=float,
                        help='model initialization range')
    parser.add_argument('-device', default='cpu', type=str,
                        help='The device. None, cpu, cuda, cuda/cpu:rank')

    opt = parser.parse_args()

    trainer = Trainer(opt)
    trainer.start_training()
