""" Get the IWSLT dataset
"""
from os.path import dirname, join
import os
from torchtext.datasets.translation import IWSLT
from torchtext import data
from subprocess import call
from ld_research.pretrain.settings import ROOT_CORPUS_DIR, PYTHONBIN, LEARN_BPE_PYTHON, APPLY_BPE_PYTHON, \
    ROOT_PT_DIR, ROOT_DATA_DIR, SRC_LANG, TGT_LANG

# Task setup
CORPUS_DIR = join(ROOT_CORPUS_DIR, 'iwslt', SRC_LANG[1:] + '-' + TGT_LANG[1:])

# From annotated transformer
# Config for moses tokenizer
MAX_LEN = 100   # Discard too long sentence
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MIN_FREQ = 1    # Discard too low freq words

# Prefix for train, valid, test
TRAIN_PREFIX = 'train'
VALID_PREFIX = 'IWSLT16.TED.tst2013'
TEST_PREFIX = 'IWSLT16.TED.tst2014'
EXTS = '.' + SRC_LANG[1:] + '-' + TGT_LANG[1:]    # '.en-de'
SRC_SUFFIX = EXTS + SRC_LANG
TGT_SUFFIX = EXTS + TGT_LANG

# path in corpus
CORPUS_TRAIN_SRC = join(CORPUS_DIR,  TRAIN_PREFIX + SRC_SUFFIX)
CORPUS_VALID_SRC = join(CORPUS_DIR, VALID_PREFIX + SRC_SUFFIX)
CORPUS_TEST_SRC = join(CORPUS_DIR, VALID_PREFIX + SRC_SUFFIX)
CORPUS_TRAIN_TGT = join(CORPUS_DIR, TRAIN_PREFIX + TGT_SUFFIX)
CORPUS_VALID_TGT = join(CORPUS_DIR, VALID_PREFIX + TGT_SUFFIX)
CORPUS_TEST_TGT = join(CORPUS_DIR, VALID_PREFIX + TGT_SUFFIX)

# BPE Setup
BPE_OPS = 10000

# BPE output
DATA_DIR = join(ROOT_DATA_DIR, 'iwslt', SRC_LANG[1:] + '-' + TGT_LANG[1:])
BPE_CODES_SRC = join(DATA_DIR, 'bpe_codes.src')
BPE_CODES_TGT = join(DATA_DIR, 'bpe_codes.tgt')
OUT_TRAIN_SRC = join(DATA_DIR, 'train.src')
OUT_VALID_SRC = join(DATA_DIR, 'valid.src')
OUT_TEST_SRC = join(DATA_DIR, 'test.src')
OUT_TRAIN_TGT = join(DATA_DIR, 'train.tgt')
OUT_VALID_TGT = join(DATA_DIR, 'valid.tgt')
OUT_TEST_TGT = join(DATA_DIR, 'test.tgt')

# Final Output
PT_DIR = join(ROOT_PT_DIR, 'iwslt', SRC_LANG[1:] + '-' + TGT_LANG[1:])
PT_PREFIX = join(PT_DIR, 'data')
OMNT_PREPROCESS = join(dirname(dirname(__file__)), 'preprocess.py')

def _download_corpus_and_moses():
    """ Get IWSLT corpus with moses tokenized """
    if not os.path.exists(CORPUS_DIR):
        print('Downloading and tokenizing {}'.format(CORPUS_DIR))
        src = data.Field(tokenize='moses', pad_token=BLANK_WORD, lower=True)
        tgt = data.Field(tokenize='moses', init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD,
                         lower=True)
        filter_pred = lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
        IWSLT.splits(exts=(SRC_LANG, TGT_LANG),
                     fields=(src, tgt),
                     filter_pred=filter_pred,
                     root=ROOT_CORPUS_DIR)
    else:
        print('Found {}, skipping...'.format(CORPUS_DIR))

def _bpe_preprocess():
    """ Apply the bpe """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

        # Learn BPE on src/tgt
        with open(BPE_CODES_SRC, 'wb') as bpe_src_stdout, \
                open(BPE_CODES_TGT, 'wb') as bpe_tgt_stdout:
            base_cmd = [PYTHONBIN, LEARN_BPE_PYTHON,
                        '-s', str(BPE_OPS),
                        '-i']
            print('Learn BPE on source...')
            call(base_cmd + [CORPUS_TRAIN_SRC], stdout=bpe_src_stdout)
            print('Learn BPE on target...')
            call(base_cmd + [CORPUS_TRAIN_TGT], stdout=bpe_tgt_stdout)

        # Apply BPE src
        with open(OUT_TRAIN_SRC, 'wb') as train,\
                open(OUT_VALID_SRC, 'wb') as valid,\
                    open(OUT_TEST_SRC, 'wb') as test:
            print('Apply BPE to source...')
            base_command = [PYTHONBIN, APPLY_BPE_PYTHON,
                            '-c', BPE_CODES_SRC, '-i']
            call(base_command + [CORPUS_TRAIN_SRC], stdout=train)
            call(base_command + [CORPUS_VALID_SRC], stdout=valid)
            call(base_command + [CORPUS_TEST_SRC], stdout=test)

        # Apply BPE tgt
        with open(OUT_TRAIN_TGT, 'wb') as train,\
                open(OUT_VALID_TGT, 'wb') as valid,\
                    open(OUT_TEST_TGT, 'wb') as test:
            print('Apply BPE to target...')
            base_command = [PYTHONBIN, APPLY_BPE_PYTHON,
                            '-c', BPE_CODES_TGT, '-i']
            call(base_command + [CORPUS_TRAIN_TGT], stdout=train)
            call(base_command + [CORPUS_VALID_TGT], stdout=valid)
            call(base_command + [CORPUS_TEST_TGT], stdout=test)
    else:
        print('Found {}, skipping...'.format(DATA_DIR))

def _omnt_preprocess():
    """ Apply OMNT preprocess """
    if not os.path.exists(PT_DIR):
        print('building pt files in {}...'.format(PT_DIR))
        os.makedirs(PT_DIR)
        cmd = [PYTHONBIN, OMNT_PREPROCESS,
               '-train_src', OUT_TRAIN_SRC,
               '-train_tgt', OUT_TRAIN_TGT,
               '-valid_src', OUT_VALID_SRC,
               '-valid_tgt', OUT_VALID_TGT,
               '-save_data', PT_PREFIX,
               ]
        call(cmd)
    else:
        print('Found {}, skipping...'.format(PT_DIR))

if __name__ == '__main__':
    """ main logic """
    _download_corpus_and_moses()
    _bpe_preprocess()
    _omnt_preprocess()
