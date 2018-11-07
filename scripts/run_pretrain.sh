#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_PATH=$(dirname ${SCRIPT_PATH})

# Switch to root path
cd ${ROOT_PATH}
GIT_COMMIT=$(git rev-parse --short=7 HEAD)
PRETRAIN=${ROOT_PATH}/pretrain.py

# Pretrain
SRC_LANG="fr"
TGT_LANG="en"

# SAVE_DIR
ROOT_SAVE_DIR=${ROOT_PATH}
EXP_NAME="exp_pretrain_${SRC_LANG}_${TGT_LANG}"
SAVE_DIR=${ROOT_SAVE_DIR}/${EXP_NAME}_${GIT_COMMIT}

# Running
python $PRETRAIN \
    -src_lang .${SRC_LANG} \
    -tgt_lang .${TGT_LANG} \
    -save_dir ${SAVE_DIR}
    # To be continued
