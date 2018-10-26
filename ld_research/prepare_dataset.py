""" prepare the ISWIT and Multi30k task1 dataset.
"""
from os.path import join
import os
from itertools import product
from settings import ROOT_CORPUS_DIR, FR, EN, DE, LOGGER, ROOT_TOK_DIR
from torchtext.datasets import IWSLT
from torchtext.data import get_tokenizer
from tqdm import tqdm
from subprocess import call

def _IWSLT_download_helper(src_lang, tgt_lang):
    """ Download result given source and target language """
    corpus_dir = join(ROOT_CORPUS_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(corpus_dir):
        LOGGER.info('iwslt {}-{} exists, skipping...'.format(src_lang[1:], tgt_lang[1:], corpus_dir))
        return
    LOGGER.info('downloading in {}...'.format(corpus_dir))
    IWSLT.dirname = IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:])
    IWSLT.urls = [IWSLT.base_url.format(src_lang[1:], tgt_lang[1:], IWSLT.dirname)]
    IWSLT.download(root=ROOT_CORPUS_DIR, check=corpus_dir)
    IWSLT.clean(corpus_dir)

def _tokenize(in_file, out_file):
    """ Use moses to tokenize the file.
        :param in_file: a str, path to a file
        :param out_file: The output file
    """
    moses_tokenzer = get_tokenizer('moses')
    with open(out_file, 'w') as out, \
        open(in_file, 'r') as inp:
        LOGGER.info('tokenizing {}...'.format(in_file))
        lines = inp.readlines()
        for line in tqdm(lines):
            tokenized_line = moses_tokenzer(line.lower())
            out.write(' '.join(tokenized_line + ['\n']))

def _tokenize_IWSLT_helper(src_lang, tgt_lang):
    """ Tokenize one of the IWSLT """
    token_dir = join(ROOT_TOK_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(token_dir):
        LOGGER.info('{} exists, skipping...'.format(token_dir))
        return
    os.makedirs(token_dir)
    corpus_dir = join(ROOT_CORPUS_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))

    # Get all suffix
    suffixs = [src_lang[1:] + '-' + tgt_lang[1:] + src_lang,
               src_lang[1:] + '-' + tgt_lang[1:] + tgt_lang]

    # Get all prefix
    prefixs = ['train', 'IWSLT16.TED.tst2013', 'IWSLT16.TED.tst2014']

    for prefix, suffix in product(prefixs, suffixs):
        in_file = join(corpus_dir, prefix + '.' + suffix)
        out_file = join(token_dir, prefix + '.' + suffix)
        _tokenize(in_file=in_file, out_file=out_file)

def _download_multi30k():
    """ Get the corpus of multi30k task1 """
    corpus_dir = join(ROOT_CORPUS_DIR, 'multi30k')
    if os.path.exists(corpus_dir):
        LOGGER.info('multi30k exists, skipping...')
        return
    LOGGER.info('Downloading multi30k task1...')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    base_url = 'https://github.com/multi30k/dataset/raw/master/data/task1/raw/{}{}.gz'
    for prefix, lang in product(prefixs, langs):
        wget_cmd = ['wget', base_url.format(prefix, lang), '-P', corpus_dir]
        call(wget_cmd)
        call(['gunzip', '-k', join(corpus_dir, '{}{}.gz'.format(prefix, lang))])

def prepare_IWSLT():
    """ Download and tokenize IWSLT """
    _IWSLT_download_helper(FR, EN)
    _IWSLT_download_helper(EN, DE)
    _tokenize_IWSLT_helper(FR, EN)
    _tokenize_IWSLT_helper(EN, DE)

def prepare_multi30k():
    """ Download and tokenize multi30k task1 """
    _download_multi30k()

    # tokenize
    corpus_dir = join(ROOT_CORPUS_DIR, 'multi30k')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    tok_dir = join(ROOT_TOK_DIR, 'multi30k')
    if os.path.exists(tok_dir):
        LOGGER.info('multi30k tokens exists, skipping...')
        return
    LOGGER.info('Tokenizing multi30k task1...')
    os.makedirs(tok_dir)
    for prefix, lang in product(prefixs, langs):
        file_name = '{}{}'.format(prefix, lang)
        in_file = join(corpus_dir, file_name)
        out_file = join(tok_dir, file_name)
        _tokenize(in_file, out_file)

if __name__ == '__main__':
    """ Main logic """
    # Get corpus and tokenize
    prepare_IWSLT()
    prepare_multi30k()

    # BPE and Get Vocab
    lang_files = {EN: join(ROOT_TOK_DIR, 'iwslt', 'en-de', 'train.en-de.en'),
                  DE: join(ROOT_TOK_DIR, 'iwslt', 'en-de', 'train.en-de.de'),
                  FR: join(ROOT_TOK_DIR, 'iwslt', 'fr-en', 'train.fr-en.fr')}

