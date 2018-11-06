""" Contains multi30k results
"""
from torch.utils.data import Dataset, DataLoader
from os.path import join
import os
import torch

from ld_research.text.utils import Vocab, pad_to_same_length
from ld_research.settings import FR, EN, DE, ROOT_BPE_DIR, EOS_WORD, BOS_WORD

# List all language
ALL_LANG = [FR, EN, DE]


class Multi30KExample:
    """ A data structure """
    def __init__(self, fr, en, de,
                 fr_lengths=0,
                 en_lengths=0,
                 de_lengths=0):
        """ A constructor """
        self.id_dicts = {FR: fr,
                         EN: en,
                         DE: de}
        self.length_dicts = {FR: fr_lengths,
                             EN: en_lengths,
                             DE: de_lengths}

    def to(self, **kwargs):
        """ Change device """
        for lang in ALL_LANG:
            self.id_dicts[lang].to(**kwargs)
            self.length_dicts[lang].to(**kwargs)

    @classmethod
    def from_dicts(cls, id_dicts, length_dicts):
        """ Get an instance from dicts """
        return cls(fr=id_dicts[FR],
                   en=id_dicts[EN],
                   de=id_dicts[DE],
                   fr_lengths=length_dicts[FR],
                   en_lengths=length_dicts[EN],
                   de_lengths=length_dicts[DE])

    """
    Some getters
    """
    @property
    def fr(self):
        return self.id_dicts[FR]

    @property
    def en(self):
        return self.id_dicts[EN]    \

    @property
    def de(self):
        return self.id_dicts[DE]

    @property
    def fr_lengths(self):
        return self.length_dicts[FR]

    @property
    def en_lengths(self):
        return self.length_dicts[EN]    \

    @property
    def de_lengths(self):
        return self.length_dicts[DE]


class Multi30KDataset(Dataset):
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
        self.padding_fns = {FR: lambda sents: pad_to_same_length(sents, EOS_WORD, None, EOS_WORD),
                            EN: lambda sents: pad_to_same_length(sents, EOS_WORD, BOS_WORD, EOS_WORD),
                            DE: lambda sents: pad_to_same_length(sents, EOS_WORD, BOS_WORD, EOS_WORD)}

    def collate_fn(self, batch):
        """ Given a list merge into one data """
        id_dicts = dict()
        length_dicts = dict()
        for lang in ALL_LANG:
            ids, lengths = self.padding_fns[lang]([example.id_dicts[lang] for example in batch])
            id_dicts[lang] = torch.tensor(self.vocabs[lang].numerize(ids)).long()
            length_dicts[lang] = torch.tensor(lengths)
        return Multi30KExample.from_dicts(id_dicts=id_dicts, length_dicts=length_dicts)
