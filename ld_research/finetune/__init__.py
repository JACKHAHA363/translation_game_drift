""" Finetune on Multi-30
"""
from torchtext.datasets import Multi30k
from torchtext import data
import os

# From annotated transformer
# Config for moses tokenizer
MAX_LEN = 100   # Discard too long sentence
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MIN_FREQ = 1    # Discard too low freq words

def download_corpus_and_moses(src_lang, tgt_lang):
    """ Get IWSLT corpus with moses tokenized at `corpus_dir` with corresponding language """
    corpus_dir = './debug'
    if not os.path.exists(corpus_dir):
        print('Downloading and tokenizing {}'.format(corpus_dir))
        src = data.Field(tokenize='moses', pad_token=BLANK_WORD, lower=True)
        tgt = data.Field(tokenize='moses', init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD,
                         lower=True)
        filter_pred = lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
        Multi30k.splits(exts=(src_lang, tgt_lang),
                        fields=(src, tgt),
                        filter_pred=filter_pred,
                        root='./corpus')
    else:
        print('Found {}, skipping...'.format(corpus_dir))

if __name__ == '__main__':
    download_corpus_and_moses('.en', 'de')