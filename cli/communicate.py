""" Pretrain the agent on IWSLT
"""
from ld_research import CommuTrainer, get_communicate_parser

if __name__ == '__main__':
    """ Parse from command """
    parser = get_communicate_parser()
    opt = parser.parse_args()
    trainer = CommuTrainer(opt)
    trainer.start_training()
