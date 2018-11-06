""" Pretrain the agent on IWSLT
"""
from ld_research import IWSLTTrainer, get_pretrain_parser

if __name__ == '__main__':
    """ Parse from command """
    parser = get_pretrain_parser()
    opt = parser.parse_args()
    trainer = IWSLTTrainer(opt)
    trainer.start_training()
