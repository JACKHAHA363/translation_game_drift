""" Get the IWSLT dataset. Both fr->en, en->fr
"""
from os.path import dirname, join
from itertools import product
import os
from torchtext.datasets.translation import IWSLT
from torchtext import data
from subprocess import call
from ld_research.pretrain.settings import ROOT_CORPUS_DIR, PYTHONBIN, LEARN_BPE_PYTHON, APPLY_BPE_PYTHON, \
    ROOT_DATA_DIR, OMNT_PREPROCESS

# Task setup
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

# Prefix for train, valid, test
TRAIN_PREFIX = 'train'
VALID_PREFIX = 'IWSLT16.TED.tst2013'
TEST_PREFIX = 'IWSLT16.TED.tst2014'

# BPE Setup
BPE_OPS = 10000

def _get_bpe_codes_path(lang):
    """ Return tha path of result of learn BPE """
    return join(ROOT_DATA_DIR, 'bpe_codes' + lang)

def _get_corpus_dir_path(src_lang, tgt_lang):
    """ Return the path to the corpus given language """
    return join(ROOT_CORPUS_DIR, 'iwslt', src_lang[1:] + '-' + tgt_lang[1:])

def _get_data_dir_path(src_lang, tgt_lang):
    """ Path to the directory of BPE output """
    return join(ROOT_DATA_DIR, 'iwslt', src_lang[1:] + '-' + tgt_lang[1:])

def _get_corpus_path(src_lang, tgt_lang, mode='train'):
    """ Return the path of (src_corpus, tgt_corpus)
        :param src_lang: The source language
        :param tgt_lang: The target language
        :param mode: The ['train', 'valid', 'test']
        :return (src_corpus, tgt_corpus)
    """
    exts = '.' + src_lang[1:] + '-' + tgt_lang[1:]    # '.en-de'
    src_suffix = exts + src_lang
    tgt_suffix = exts + tgt_lang

    # path in corpus
    if mode == 'train':
        mode_prefix = TRAIN_PREFIX
    elif mode == 'valid':
        mode_prefix = VALID_PREFIX
    elif mode == 'test':
        mode_prefix = TEST_PREFIX
    else:
        raise ValueError('mode {} is invalid'.format(mode))
    corpus_dir = _get_corpus_dir_path(src_lang, tgt_lang)
    src_corpus = join(corpus_dir, mode_prefix + src_suffix)
    tgt_corpus = join(corpus_dir, mode_prefix + tgt_suffix)
    return src_corpus, tgt_corpus

def _apply_BPE(in_file, out_file, lang):
    """ Apply the BPE """
    cmd = [PYTHONBIN, APPLY_BPE_PYTHON,
           '-c', _get_bpe_codes_path(lang),
           '-i', in_file]
    with open(out_file, 'w') as f:
        call(cmd, stdout=f)

#############################################################

def download_corpus_and_moses(src_lang, tgt_lang):
    """ Get IWSLT corpus with moses tokenized at `corpus_dir` with corresponding language """
    corpus_dir = _get_corpus_dir_path(src_lang, tgt_lang)
    if not os.path.exists(corpus_dir):
        print('Downloading and tokenizing {}'.format(corpus_dir))
        src = data.Field(tokenize='moses', pad_token=BLANK_WORD, lower=True)
        tgt = data.Field(tokenize='moses', init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD,
                         lower=True)
        filter_pred = lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
        IWSLT.splits(exts=(src_lang, tgt_lang),
                     fields=(src, tgt),
                     filter_pred=filter_pred,
                     root=ROOT_CORPUS_DIR)
    else:
        print('Found {}, skipping...'.format(corpus_dir))

def learn_bpe_codes(lang):
    """ Apply BPE codes for all lang """
    bpe_codes_path = join(ROOT_DATA_DIR, 'bpe_codes' + lang)
    if os.path.exists(bpe_codes_path):
        print('{} exists, skipping...'.format(bpe_codes_path))
        return

    if lang == EN:
        corpus, _ = _get_corpus_path(EN, DE, mode='train')
    elif lang == DE:
        _, corpus = _get_corpus_path(EN, DE, mode='train')
    elif lang == FR:
        corpus, _ = _get_corpus_path(FR, EN, mode='train')
    else:
        raise ValueError('Invalid language {}'.format(lang))
    print('Learn BPE on language {}'.format(lang))
    cmd = [PYTHONBIN, LEARN_BPE_PYTHON,
           '-s', str(BPE_OPS),
           '-i', corpus]
    with open(bpe_codes_path, 'wb') as f:
        call(cmd, stdout=f)

def apply_BPE(src_lang, tgt_lang):
    """ Apply the BPE """
    data_dir = _get_data_dir_path(src_lang, tgt_lang)
    if os.path.exists(data_dir):
        print('Found {}, skipping...'.format(data_dir))
        return
    os.makedirs(data_dir)

    # Apply to src
    modes = ['train', 'valid', 'test']
    for mode in modes:
        print('Apply BPE to {}->{} [{}]'.format(src_lang, tgt_lang, mode))
        corpus_src, corpus_tgt = _get_corpus_path(src_lang, tgt_lang, mode)
        src_out = join(data_dir, mode + '.src')
        tgt_out = join(data_dir, mode + '.tgt')
        _apply_BPE(in_file=corpus_src, out_file=src_out, lang=src_lang)
        _apply_BPE(in_file=corpus_tgt, out_file=tgt_out, lang=tgt_lang)

def omnt_preprocess(src_lang, tgt_lang):
    """ Apply OMNT preprocess """
    pt_dir = join(ROOT_DATA_DIR, 'pretrain', src_lang[1:] + tgt_lang[1:])
    data_dir = _get_data_dir_path(src_lang, tgt_lang)
    if not os.path.exists(pt_dir):
        print('building pt files {}...'.format(pt_dir))
        os.makedirs(pt_dir)
        cmd = [PYTHONBIN, OMNT_PREPROCESS,
               '-train_src', join(data_dir, 'train.src'),
               '-train_tgt', join(data_dir, 'train.tgt'),
               '-valid_src', join(data_dir, 'valid.src'),
               '-valid_tgt', join(data_dir, 'valid.tgt'),
               '-save_data', join(pt_dir, 'data'),
               ]
        call(cmd)
    else:
        print('Found {}, skipping...'.format(pt_dir))

if __name__ == '__main__':
    """ main logic """
    # Download both corpus
    download_corpus_and_moses(src_lang=FR, tgt_lang=EN)
    download_corpus_and_moses(src_lang=EN, tgt_lang=DE)

    # Learn BPE codes
    learn_bpe_codes(FR)
    learn_bpe_codes(EN)
    learn_bpe_codes(DE)

    # Apply it
    apply_BPE(FR, EN)
    apply_BPE(EN, DE)


    # OMNT preprocess to .pt
    omnt_preprocess(FR, EN)
    omnt_preprocess(EN, DE)
