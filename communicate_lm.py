""" Pretrain the agent on IWSLT
"""
import torch

from ld_research import CommuLMTrainer, get_communicate_lm_parser
from ld_research.model import LanguageModel

def get_agent_state_dict(ckpt_path):
    """ Get agent state_dict from either checkpoint """
    ckpt = torch.load(ckpt_path,
                      map_location=lambda storage, loc: storage)
    model_state_dict = ckpt.get('agent')
    if model_state_dict is None:
        raise ValueError('Something wrong with checkpoint')
    return model_state_dict


if __name__ == '__main__':
    """ Parse from command """
    parser = get_communicate_lm_parser()
    opt = parser.parse_args()

    # Get language model
    lm = LanguageModel()
    state_dict = get_agent_state_dict(ckpt_path=opt.lm_ckpt)
    lm.load_state_dict(state_dict)

    # Train
    trainer = CommuLMTrainer(opt, lm)
    trainer.start_training()
