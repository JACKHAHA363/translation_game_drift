""" Contains multi30k results
"""
from torch.utils.data import Dataset
from os.path import join
import os

from ld_research.text.utils import Vocab
from ld_research.settings import FR, EN, DE, ROOT_BPE_DIR

# List all language
ALL_LANG = [FR, EN, DE]


class Multi30KExample:
    """ A data structure """
    def __init__(self, fr, fr_lengths, en, en_lengths, de, de_lengths):
        """ A constructor """
        self.fr = fr
        self.fr_lengths = fr_lengths
        self.en = en
        self.en_lengths = en_lengths
        self.de = de
        self.de_lengths = de_lengths

    def to(self, **kwargs):
        """ Change device """
        self.fr = self.fr.to(**kwargs)
        self.en = self.de.to(**kwargs)
        self.de = self.de.to(**kwargs)
        self.fr_lengths = self.fr_lengths.to(**kwargs)
        self.en_lengths = self.en_lengths.to(**kwargs)
        self.de_lengths = self.de_lengths.to(**kwargs)


class Multi30K(Dataset):
    """ A dataset object for Multi30k """
    prefix = {'train': 'train',
              'valid': 'val',
              'test': 'test_2017_flickr'}

    def __init__(self, mode='train'):
        """ Constructor """
        assert mode in ['train', 'test', 'valid'], "Invalid mode {}".format(mode)
        self.mode = mode
        self.vocabs = {lang: Vocab(src_lang=lang) for lang in ALL_LANG}
        self.texts = {lang: self._get_txt(lang, mode) for lang in ALL_LANG}
        assert len(self.texts[EN]) == len(self.texts[DE])
        assert len(self.texts[EN]) == len(self.texts[FR])

    def get_txt(self, lang, mode='train'):
        """ Return the txt of that language """
        bpe_corpus_dir = join(ROOT_BPE_DIR, 'multi30k')
        txt_name = join(bpe_corpus_dir, self.prefix[mode] + lang)
        with open(txt_name, 'r', encoding='utf-8') as file:
            contents = file.readlines()
        return contents

    def __len__(self):
        """ length """
        return len(self.texts[EN])

    def __getitem__(self, index):
        """ Return a tripple """


