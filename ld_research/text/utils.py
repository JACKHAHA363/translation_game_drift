""" Data preprocessing loading utility
"""
from os.path import join, basename
import os
from itertools import product
from torchtext.datasets import IWSLT
from torchtext.data import get_tokenizer
from tqdm import tqdm
from subprocess import call

from ld_research.settings import ROOT_CORPUS_DIR, FR, EN, DE, LOGGER, ROOT_TOK_DIR, ROOT_BPE_DIR, \
    LEARN_JOINT_BPE, APPLY_BPE, PYTHONBIN, MIN_FREQ, BOS_WORD, EOS_WORD, UNK_WORD, PAD_WORD


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
        LOGGER.info('tokenizing {}...'.format(basename(in_file)))
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


def learn_bpe():
    """ Learn the BPE and get vocab """
    if not os.path.exists(ROOT_BPE_DIR):
        os.makedirs(ROOT_BPE_DIR)
    lang_files = {EN: join(ROOT_TOK_DIR, 'iwslt', 'en-de', 'train.en-de.en'),
                  DE: join(ROOT_TOK_DIR, 'iwslt', 'en-de', 'train.en-de.de'),
                  FR: join(ROOT_TOK_DIR, 'iwslt', 'fr-en', 'train.fr-en.fr')}

    # BPE and Get Vocab
    if not os.path.exists(join(ROOT_BPE_DIR, 'bpe.codes')):
        learn_bpe_cmd = [PYTHONBIN, LEARN_JOINT_BPE]
        learn_bpe_cmd += ['--input'] + [lang_files[lang] for lang in lang_files.keys()]
        learn_bpe_cmd += ['-s', '10000']
        learn_bpe_cmd += ['-o', join(ROOT_BPE_DIR, 'bpe.codes')]
        learn_bpe_cmd += ['--write-vocabulary'] + [join(ROOT_BPE_DIR, 'vocab' + lang)
                                                   for lang in lang_files.keys()]
        LOGGER.info('Learning BPE on joint language...')
        call(learn_bpe_cmd)
    else:
        LOGGER.info('bpe.codes file exist, skipping...')

def apply_bpe(in_file, out_file, lang):
    """ Apply BPE """
    codes_file = join(ROOT_BPE_DIR, 'bpe.codes')
    assert os.path.exists(codes_file), '{} not exists!'.format(codes_file)
    vocab_file = join(ROOT_BPE_DIR, 'vocab' + lang)
    cmd = [PYTHONBIN, APPLY_BPE]
    cmd += ['-c', codes_file]
    cmd += ['--vocabulary', vocab_file]
    cmd += ['--vocabulary-threshold', str(MIN_FREQ)]
    cmd += ['--input', in_file]
    cmd += ['--output', out_file]
    LOGGER.info('Applying BPE to {}'.format(basename(out_file)))
    call(cmd)


def apply_bpe_iwslt(src_lang, tgt_lang):
    """ Apply BPE to iwslt with `src_lang` and `tgt_lang` """
    bpe_dir = join(ROOT_BPE_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(bpe_dir):
        LOGGER.info('BPE IWSLT for {}-{} exists, skipping...'.format(src_lang[1:], tgt_lang[1:]))
        return
    os.makedirs(bpe_dir)
    tok_dir = join(ROOT_TOK_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    suffixs = [src_lang[1:] + '-' + tgt_lang[1:] + src_lang,
               src_lang[1:] + '-' + tgt_lang[1:] + tgt_lang]
    prefixs = ['train', 'IWSLT16.TED.tst2013', 'IWSLT16.TED.tst2014']
    for prefix, suffix in product(prefixs, suffixs):
        tokenized_file = join(tok_dir, prefix + '.' + suffix)
        bpe_out = join(bpe_dir, prefix + '.' + suffix)
        apply_bpe(in_file=tokenized_file, out_file=bpe_out, lang=suffix[-3:])


def apply_bpe_multi30k():
    """ Apply BPE to multi30k """
    bpe_dir = join(ROOT_BPE_DIR, 'multi30k')
    if os.path.exists(bpe_dir):
        LOGGER.info('BPE Multi30k exists, skipping...')
        return
    os.makedirs(bpe_dir)
    tok_dir = join(ROOT_TOK_DIR, 'multi30k')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    for prefix, lang in product(prefixs, langs):
        file_name = prefix + lang
        in_file = join(tok_dir, file_name)
        out_file = join(bpe_dir, file_name)
        apply_bpe(in_file, out_file, lang=lang)


def get_vocab_file(lang):
    """ Return the vocab file of a language """
    return join(ROOT_BPE_DIR, 'vocab' + lang)


class Vocab:
    """ the vocab object """
    def __init__(self, lang=None, words_with_freq=None):
        """ constructor """
        if words_with_freq:
            self.words_with_freq = words_with_freq
        elif lang:
            vocab_file = get_vocab_file(lang)
            self.words_with_freq = dict()
            with open(vocab_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    word, freq = line.rstrip('\n').split()
                    self.words_with_freq[word] = freq
        else:
            raise ValueError('lang and words_with_freq can not be both None')

        # Add utlity tokens
        self.words_with_freq[BOS_WORD] = 1
        self.words_with_freq[EOS_WORD] = 1
        self.words_with_freq[UNK_WORD] = 1
        self.words_with_freq[PAD_WORD] = 1

        # get words
        self.idx2words = list(self.words_with_freq.keys())
        self.words2idx = {word: self.idx2words.index(word)
                          for word in self.idx2words}

    def __len__(self):
        """ Return total number of words """
        return len(self.idx2words)

    def get_index(self, word):
        """ Return the index of a word """
        idx = self.words2idx.get(word)
        if idx is None:
            LOGGER.warning('{} not in vocabulary replace with UNK'.format(word))
            idx = self.words2idx.get(UNK_WORD)
        return idx

    def get_word(self, index):
        """ Return the word """
        return self.idx2words[index]

    def numerize(self, data):
        """ Numerize the example with the vocab. Return a torch Tensor
            by unfolding the structure recursively.
        """
        if type(data) is str:
            return self.get_index(data)
        elif type(data) is list:
            result = []
            for element in data:
                result.append(self.numerize(element))
            return result
        else:
            raise ValueError('Unkown type {}'.format(type(data)))

    def denumerize(self, data):
        """ The opposite of numerize """
        if type(data) is int:
            return self.get_word(data)
        elif type(data) is list:
            result = []
            for element in data:
                result.append(self.denumerize(element))
            return result
        else:
            raise ValueError('Unkown type {}'.format(type(data)))

    def to_sentences(self, ids, excludes=None):
        """ Get a list of list of words and exclude some of them
            :param ids: list of list of word index (int).
            :param excludes: List of words to exclude. If None default to BOS, EOS, PAD
            :return: sentences. A list of list of str.
        """
        excludes = excludes if excludes else [BOS_WORD, EOS_WORD, PAD_WORD]
        sentences = []
        for curr_ids in ids:
            curr_words = self.denumerize(curr_ids)
            curr_words = [word for word in curr_words if word not in excludes]
            sentences += [curr_words]
        return sentences


def pad_to_same_length(sentences, pad_token=PAD_WORD,
                       init_token=None, end_token=None):
    """ Given a list of sentences. Pad each to the maximum length.
        :param sentences: A list of list of indices
        :param pad_token: The padding token
        :param init_token: The initial token. If None don't pad
        :param end_token: The ending token. If None don't pad
        :return (results, lenghs). The padded sentences along with the lengths.
    """
    max_len = max([len(sentence) for sentence in sentences])
    results = []
    lengths = []
    for sentence in sentences:
        # Beginning
        padded, length = [], 0
        if init_token:
            padded += [init_token]
            length += 1

        # Original sentence
        padded += sentence
        length += len(sentence)

        # End of sentence
        if end_token:
            padded += [end_token]
            length += 1

        # Padding
        padded += [pad_token] * (max_len - len(sentence))

        # Add to results
        results += [padded]
        lengths += [length]
    return results, lengths
