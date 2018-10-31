"""
    Created by yuchen on 10/31/18
    Description:
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader

from settings import ROOT_BPE_DIR, PAD_WORD, EOS_WORD, BOS_WORD
from text import Vocab


class IWSLTDataset(Dataset):
    """ An object for dataset """
    prefix = {'train': 'train',
              'valid': 'IWSLT16.TED.tst2013',
              'test': 'IWSLT16.TED.tst2014'}

    def __init__(self, src_lang, tgt_lang, mode='train'):
        """ constructor
            :param src_lang: The src language
            :param tgt_lang: The target language
            :param mode: The train, valid, test mode
        """
        assert mode in ['train', 'test', 'valid'], "Invalid mode {}".format(mode)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.mode = mode
        self.src_vocab = Vocab(lang=src_lang)
        self.tgt_vocab = Vocab(lang=tgt_lang)
        self.bpe_corpus_dir = os.path.join(ROOT_BPE_DIR, 'iwslt',
                                           '{}-{}'.format(src_lang[1:], tgt_lang[1:]))
        self.src_text, self.tgt_text = self.get_txt(mode)
        assert len(self.src_text) == len(self.tgt_text), "Not paired dataset, something is wrong"

    def get_txt(self, mode='train'):
        """ Return the txt """
        base_text_name = '{}.' + self.src_lang[1:] + '-' \
                         + self.tgt_lang[1:]
        base_text_name_src = base_text_name + self.src_lang
        base_text_name_tgt = base_text_name + self.tgt_lang
        src_txt = os.path.join(self.bpe_corpus_dir,
                               base_text_name_src.format(self.prefix[mode]))
        tgt_txt = os.path.join(self.bpe_corpus_dir,
                               base_text_name_tgt.format(self.prefix[mode]))
        with open(src_txt, 'r', encoding='utf-8') as src_file, \
                open(tgt_txt, 'r', encoding='utf-8') as tgt_file:
            src_contents = src_file.readlines()
            tgt_contents = tgt_file.readlines()
        return src_contents, tgt_contents

    def __len__(self):
        """ return the length """
        return len(self.src_text)

    def __getitem__(self, index):
        """ Return a src and tgt sentence """
        src_sentence = self.src_text[index].rstrip('\n').split()
        tgt_sentence = self.tgt_text[index].rstrip('\n').split()
        return IWSLTExample(src=src_sentence, tgt=tgt_sentence, src_lengths=0, tgt_lengths=0)


class IWSLTExample:
    """ A data structure """
    def __init__(self, src, tgt, src_lengths, tgt_lengths):
        """ A constructor """
        self.src = src
        self.tgt = tgt
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths

    def to(self, **kwargs):
        """ Change device """
        self.src = self.src.to(**kwargs)
        self.tgt = self.tgt.to(**kwargs)
        self.src_lengths = self.src_lengths.to(**kwargs)
        self.tgt_lengths = self.tgt_lengths.to(**kwargs)


class IWSLTDataloader(DataLoader):
    """ My dataloader """
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        """ Constructor """
        super(IWSLTDataloader, self).__init__(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=None,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              timeout=timeout,
                                              worker_init_fn=worker_init_fn,
                                              collate_fn=self.collate_fn,
                                              drop_last=drop_last)
        self.src_vocab = self.dataset.src_vocab
        self.tgt_vocab = self.dataset.tgt_vocab

    def _pad_to_same_length(self, sentences, pad_token=PAD_WORD,
                            init_token=None, end_token=None):
        """ Given a list of sentences. Pad each to the maximum length.
            :param sentences: A list of list of indices
            :param pad_token: The padding token
            :param init_token: The initial token. If None don't pad
            :param end_token: The ending token. If None don't pad
            :return (results, lenghs). The padded sentences along with the lengths./
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

    def collate_fn(self, batch):
        """ Merge a list of IWSLTExample into one IWSLTExample """
        src, src_lengths = self._pad_to_same_length(sentences=[example.src for example in batch],
                                                    pad_token=PAD_WORD, init_token=None,
                                                    end_token=EOS_WORD)
        tgt, tgt_lengths = self._pad_to_same_length(sentences=[example.tgt for example in batch],
                                                    pad_token=PAD_WORD, init_token=BOS_WORD,
                                                    end_token=EOS_WORD)
        src = self._to_tensor(self.src_vocab.numerize(src))
        tgt = self._to_tensor(self.tgt_vocab.numerize(tgt))
        src_lengths = self._to_tensor(src_lengths)
        tgt_lengths = self._to_tensor(tgt_lengths)
        return IWSLTExample(src=src, src_lengths=src_lengths,
                            tgt=tgt, tgt_lengths=tgt_lengths)

    def _to_tensor(self, input_list):
        """ Turn a list to tensor """
        return torch.tensor(input_list,
                            dtype=torch.int64)