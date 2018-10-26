""" Some settings for both finetune and pretrain
"""
from os.path import dirname, join
import logging
import os

# Language
FR = '.fr'
EN = '.en'
DE = '.de'

# From annotated transformer
# Config for moses tokenizer
MAX_LEN = 100   # Discard too long sentence
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MIN_FREQ = 1    # Discard too low freq words

def _maybe_makedirs(path_to_dir):
    """ Create dir if necessary """
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

# PYTHONBIN (change to yours)
#PYTHONBIN = '/u/luyuchen/miniconda2/envs/pytorch/bin/python'
PYTHONBIN = '/home/yuchen/miniconda3/envs/pommerman/bin/python'

# Some Paths
LD_RESEARCH_FOLDER = dirname(__file__)
LD_FOLDER = dirname(LD_RESEARCH_FOLDER)

# Store data part
DATA_FOLDER = join(LD_FOLDER, 'data')
_maybe_makedirs(DATA_FOLDER)

# Store the raw corpus
ROOT_CORPUS_DIR = join(DATA_FOLDER, 'corpus')
_maybe_makedirs(ROOT_CORPUS_DIR)

# Store the tokenize result
ROOT_TOK_DIR = join(DATA_FOLDER, 'tok')
_maybe_makedirs(LD_FOLDER)

# Store BPE results
ROOT_BPE_DIR = join(DATA_FOLDER, 'bpe')

# BPE Setup
TOOL_DIR = join(LD_FOLDER, 'tools')
LEARN_BPE_PYTHON = join(TOOL_DIR, 'learn_bpe.py')
APPLY_BPE_PYTHON = join(TOOL_DIR, 'apply_bpe.py')
OMNT_PREPROCESS = join(LD_FOLDER, 'preprocess.py')

# Storing results
ROOT_RESULT_FOLDER = join(LD_FOLDER, 'runs')

# LOGGER
LOGGER = logging.getLogger()

def config_logger():
    """ Config the logger """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    LOGGER.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    LOGGER.handlers = [console_handler]
config_logger()
