""" Contains multi30k results
"""
from torch.utils.data import Dataset, DataLoader
from os.path import join
import os

from ld_research.text.utils import Vocab, pad_to_same_length
from ld_research.settings import FR, EN, DE, ROOT_BPE_DIR, PAD_WORD, EOS_WORD, BOS_WORD

# List all language
ALL_LANG = [FR, EN, DE]


class Multi30KExample:
    """ A data structure """
    def __init__(self, fr, en, de,
                 fr_lengths=0,
                 en_lengths=0,
                 de_lengths=0):
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
        self.vocabs = {lang: Vocab(lang) for lang in ALL_LANG}
        self.texts = {lang: self._get_txt(lang, mode) for lang in ALL_LANG}
        assert len(self.texts[EN]) == len(self.texts[DE])
        assert len(self.texts[EN]) == len(self.texts[FR])

    def _get_txt(self, lang, mode='train'):
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
        """ Return a triple """
        fr = self.texts[FR][index].rstrip('\n').split()
        en = self.texts[EN][index].rstrip('\n').split()
        de = self.texts[DE][index].rstrip('\n').split()
        return Multi30KExample(fr=fr, en=en, de=de)


class Multi30KLoader(DataLoader):
    """ The dataloader for multi30k """
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        """ Constructor """
        super(Multi30KLoader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=None,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=self.collate_fn,
                                             drop_last=drop_last)
        self.vocabs = self.dataset.vocabs

    def collate_fn(self, batch):
        """ Given a list merge into one data """
        fr, fr_lengths = pad_to_same_length(sentences=[example.fr for example in batch],
                                            pad_token=PAD_WORD, init_token=None,
                                            end_token=EOS_WORD)
        en, en_lengths = pad_to_same_length(sentences=[example.en for example in batch],
                                            pad_token=PAD_WORD, init_token=None,
                                            end_token=None)
        de, de_lengths = pad_to_same_length(sentences=[example.de for example in batch],
                                            pad_token=PAD_WORD, init_token=BOS_WORD,
                                            end_token=EOS_WORD)
        return Multi30KExample(fr=fr, en=en, de=de, fr_lengths=fr_lengths,
                               en_lengths=en_lengths, de_lengths=de_lengths)
