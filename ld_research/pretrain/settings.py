""" Global config for pretrain
"""
from os.path import dirname, join
import os
from datetime import datetime
from ld_research.settings import *

# Result for pretrain results
PRETRAIN_RESULTS_FOLDER = join(ROOT_RESULT_FOLDER, 'pretrain')

# Experiment
SRC_LANG = '.en'    # change this
TGT_LANG = '.de'    # change this
EXP_FOLDER = join(PRETRAIN_RESULTS_FOLDER,
                  SRC_LANG[1:] + '-' + TGT_LANG[1:] + datetime.now().strftime("/%b-%d_%H-%M-%S"))
