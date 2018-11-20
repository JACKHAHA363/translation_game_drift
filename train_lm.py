""" Pretrain language model on IWSLT
"""
from ld_research import LMTrainer, get_language_model_parser

if __name__ == '__main__':
    """ Parse from command """
    parser = get_language_model_parser()
    opt = parser.parse_args()
    trainer = LMTrainer(opt)
    trainer.start_training()
