#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:00

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_PATH=$(dirname ${SCRIPT_PATH})

# Switch to root path
cd ${ROOT_PATH}
GIT_COMMIT=$(git rev-parse --short=7 HEAD)
echo "Running code ${GIT_COMMIT}"

PRETRAIN=${ROOT_PATH}/pretrain.py

# Pretrain
SRC_LANG="fr"
TGT_LANG="en"

# SAVE_DIR
ROOT_SAVE_DIR=${LANG_DRIFT}/pretrain
EXP_NAME="exp_pretrain_${SRC_LANG}_${TGT_LANG}"
SAVE_DIR=${ROOT_SAVE_DIR}/${EXP_NAME}_${GIT_COMMIT}

# Running
source activate pytorch41
python $PRETRAIN \
    -src_lang .${SRC_LANG} \
    -tgt_lang .${TGT_LANG} \
    -dropout 0.1 \
    -train_steps 200000 \
    -save_dir ${SAVE_DIR}

