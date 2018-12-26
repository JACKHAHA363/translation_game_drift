# Language Drift in Collaborative Translation Game
This is a reproduction of the paper [Countering Language Drift with Grounding](https://openreview.net/forum?id=BkMn9jAcYQ). This is not the
official implementation of the paper, but is part of the [ICLR reproducibility challange](https://reproducibility-challenge.github.io/iclr_2019/).

This repo contains the implementation for pretraining as well as policy gradient baselines. It also implements a fine-tuning method coupled with
language model and a PPO baseline, both these two haven't been made working yet.

# Installation
Clone this repo by
```
git clone https://github.com/diplomacy-game/language-drift --recursive .
```
Install `pytorch 1.0`, `NLTK`, and `torchtext`. Install the other dependency when it fits. Then run
```
pip install -e .
```

# Prepare Dataset
Run './cli/prepare_dataset'. It would put everythin under `./data` folder.

# Pretrain
Pretrain the fr-de agent and en-de agent by
```
python ./cli/pretrain.py \
	-src_lang .fr -tgt_lang .en \
	-logging_steps 100 \
	-checkpoint_steps 2000 \
	-valid_steps 2000 \
	-train_steps 100000 \
	-start_decay_steps 30000 \
	-dropout 0.1 \
	-device cuda \
	-optim adam \
	-learning_rate 0.001 \
	-adam_beta2 0.98 \
	-save_dir $SAVE_DIR
```
```
python ./cli/pretrain.py \
	-src_lang .en -tgt_lang .de \
	-logging_steps 100 \
	-checkpoint_steps 2000 \
	-valid_steps 2000 \
	-train_steps 100000 \
	-start_decay_steps 30000 \
	-dropout 0.1 \
	-device cuda \
	-optim adam \
	-learning_rate 0.001 \
	-adam_beta2 0.98 \
	-save_dir $SAVE_DIR
```

# Finetune with RL
To fine-tune with RL one can run
```
python ./cli/communicate.py \
	-pretrain_fr_en_ckpt ${PRETRAIN_FR_EN} \
	-pretrain_en_de_ckpt ${PRETRAIN_EN_DE} \
	-ent_coeff 0.01 \
	-fr_en_learning_rate 0.0001 \
	-norm_reward \
	-logging_steps 10 \
	-batch_size 256 \
	-checkpoint_steps 100 \
	-valid_steps 20 \
	-train_steps 10000 \
	-device cuda \
	-fr_en_optim adam \
	-fr_en_start_decay_steps 1000 \
	-fr_en_decay_steps 1000 \
	-fr_en_learning_rate_decay 1. \
	-fr_en_max_grad_norm 1 \
	-en_de_optim adam \
	-en_de_learning_rate 0.001 \
	-en_de_start_decay_steps 1000 \
	-en_de_decay_steps 1000 \
	-en_de_learning_rate_decay 0.5 \
	-value_optim adam \
	-value_learning_rate 0.001 \
	-value_start_decay_steps 10000 \
	-value_decay_steps 1000 \
	-dropout 0.1 \
	-save_dir $SAVE_DIR
```
Add `-reduce_ent` flag to use the opposite of entropy loss. Add `-disable_dropout` flag to not using dropout.
All flags can be found in `ld_research/parser.py`.