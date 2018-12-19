from ld_research.parser import get_communicate_parser, get_pretrain_parser, \
    get_language_model_parser, get_communicate_lm_parser, get_communicate_ppo_parser
from ld_research.training.iwslt import Trainer as IWSLTTrainer
from ld_research.training.finetune import Trainer as CommuTrainer, LMFinetune as CommuLMTrainer
from ld_research.training.language_model import Trainer as LMTrainer
from ld_research.training.finetune_ppo import Trainer as PPOCommuTrainer
