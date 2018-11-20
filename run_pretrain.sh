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

#SAVE_DIR=${LANGUAGE_DRIFT}/pretrain_fr_en_${GIT_COMMIT}_adam
#rm $SAVE_DIR -rf
#python ~/work/language-drift/cli/pretrain.py \
#	-src_lang .fr -tgt_lang .en \
#	-logging_steps 100 \
#	-checkpoint_steps 2000 \
#	-valid_steps 2000 \
#	-train_steps 100000 \
#	-start_decay_steps 30000 \
#	-dropout 0.1 \
#	-device cuda \
#	-optim adam \
#	-learning_rate 0.001 \
#	-adam_beta2 0.98 \
#	-save_dir $SAVE_DIR
#

SAVE_DIR=${LANGUAGE_DRIFT}/pretrain_en_de_${GIT_COMMIT}
rm $SAVE_DIR -rf
python ~/work/language-drift/cli/pretrain.py \
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

