""" Global config for pretrain
"""
from os.path import dirname, join
from datetime import datetime

# PYTHONBIN (change to yours)
PYTHONBIN = '/home/yuchen/miniconda3/envs/pommerman/bin/python'
SRC_LANG = '.en'    # change this
TGT_LANG = '.de'    # change this

# Some Paths
RESEARCH_FOLDER = dirname(__file__)
OMNT_FOLDER = dirname(dirname(RESEARCH_FOLDER))

# Store the actual corpus
ROOT_CORPUS_DIR = join(OMNT_FOLDER, 'corpus')

# Store the BPE and final pt files
ROOT_DATA_DIR = join(OMNT_FOLDER, 'data')
ROOT_PT_DIR = join(ROOT_DATA_DIR, 'pt_files')

# BPE Setup
TOOL_DIR = join(OMNT_FOLDER, 'tools')
LEARN_BPE_PYTHON = join(TOOL_DIR, 'learn_bpe.py')
APPLY_BPE_PYTHON = join(TOOL_DIR, 'apply_bpe.py')

# Storing results
ROOT_RESULT_FOLDER = join(OMNT_FOLDER, 'runs')
PRETRAIN_RESULTS_FOLDER = join(ROOT_RESULT_FOLDER, 'pretrain')
EXP_FOLDER = join(PRETRAIN_RESULTS_FOLDER,
                  SRC_LANG[1:] + '-' + TGT_LANG[1:] + datetime.now().strftime("/%b-%d_%H-%M-%S"))
