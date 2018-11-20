#!/bin/bash
#SBATCH --gres=gpu:p100-16gb
#SBATCH --mem=32G
#SBATCH --qos=high

source activate pytorch
export PYTHONIOENCODING="utf-8"

PROJECT_ROOT=~/work/language-drift
cd $PROJECT_ROOT
GIT_COMMIT=`git rev-parse HEAD`
GIT_COMMIT=${GIT_COMMIT:0:7}
echo ${GIT_COMMIT}

SAVE_DIR_ROOT=${LANGUAGE_DRIFT}/20181120
mkdir $SAVE_DIR_ROOT -p

SAVE_DIR=${SAVE_DIR_ROOT}/pg_${GIT_COMMIT}
rm $SAVE_DIR -rf

PRETRAIN_FR_EN=${LANGUAGE_DRIFT}/checkpoints/pretrain_fr_en.ckpt
PRETRAIN_EN_DE=${LANGUAGE_DRIFT}/checkpoints/pretrain_en_de.ckpt
python ~/work/language-drift/cli/communicate.py \
	-pretrain_fr_en_ckpt ${PRETRAIN_FR_EN} \
	-pretrain_en_de_ckpt ${PRETRAIN_EN_DE} \
	-logging_steps 10 \
	-batch_size 256 \
	-checkpoint_steps 100 \
	-valid_steps 20 \
	-train_steps 10000 \
	-device cuda \
	-ent_coeff 0.001 \
	-fr_en_optim adam \
	-fr_en_learning_rate 0.0002 \
	-fr_en_start_decay_steps 10000 \
	-fr_en_decay_steps 500 \
	-fr_en_learning_rate_decay 0.99 \
	-en_de_optim adam \
	-en_de_learning_rate 0.001 \
	-en_de_start_decay_steps 1000 \
	-en_de_decay_steps 1000 \
	-en_de_learning_rate_decay 0.5 \
	-value_optim adam \
	-value_learning_rate 0.001 \
	-value_start_decay_steps 10000 \
	-value_decay_steps 1000 \
	-sample_method greedy \
	-dropout 0.1 \
	-save_dir $SAVE_DIR

