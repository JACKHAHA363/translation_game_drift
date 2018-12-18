""" Pretrain the agent on IWSLT
"""
from ld_research import PPOCommuTrainer, get_communicate_ppo_parser

if __name__ == '__main__':
    """ Parse from command """
    parser = get_communicate_ppo_parser()
    opt = parser.parse_args()
    trainer = PPOCommuTrainer(opt)
    trainer.start_training()
