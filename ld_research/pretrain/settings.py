""" Global config for pretrain
"""
from os.path import dirname, join
import os
from datetime import datetime

def _maybe_makedirs(path_to_dir):
    """ Create dir if necessary """
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

# PYTHONBIN (change to yours)
PYTHONBIN = '/u/luyuchen/miniconda2/envs/pytorch/bin/python'
SRC_LANG = '.en'    # change this
TGT_LANG = '.de'    # change this

# Some Paths
RESEARCH_FOLDER = dirname(__file__)
OMNT_FOLDER = dirname(dirname(RESEARCH_FOLDER))

# Store the raw corpus
ROOT_CORPUS_DIR = join(OMNT_FOLDER, 'corpus')
_maybe_makedirs(ROOT_CORPUS_DIR)

# Store the dataset used for training
ROOT_DATA_DIR = join(OMNT_FOLDER, 'data')
_maybe_makedirs(ROOT_DATA_DIR)

# BPE Setup
TOOL_DIR = join(OMNT_FOLDER, 'tools')
LEARN_BPE_PYTHON = join(TOOL_DIR, 'learn_bpe.py')
APPLY_BPE_PYTHON = join(TOOL_DIR, 'apply_bpe.py')
OMNT_PREPROCESS = join(OMNT_FOLDER, 'preprocess.py')

# Storing results
ROOT_RESULT_FOLDER = join(OMNT_FOLDER, 'runs')
PRETRAIN_RESULTS_FOLDER = join(ROOT_RESULT_FOLDER, 'pretrain')
EXP_FOLDER = join(PRETRAIN_RESULTS_FOLDER,
                  SRC_LANG[1:] + '-' + TGT_LANG[1:] + datetime.now().strftime("/%b-%d_%H-%M-%S"))
