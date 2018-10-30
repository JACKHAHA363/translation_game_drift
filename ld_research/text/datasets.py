""" Contains object for dataset and vocab
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from ld_research.settings import ROOT_BPE_DIR, BOS_WORD, EOS_WORD, LOGGER, UNK_WORD, PAD_WORD
from ld_research.text.utils import get_vocab_file
import os

class Vocab:
    """ the vocab object """
    def __init__(self, lang=None, words_with_freq=None):
        """ constructor """
        if words_with_freq:
            self.words_with_freq = words_with_freq
        elif lang:
            vocab_file = get_vocab_file(lang)
            self.words_with_freq = dict()
            with open(vocab_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    word, freq = line.split()
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

class IWSLTDataset(Dataset):
    """ An object for dataset """
    prefix = {'train': 'train',
              'valid': 'IWSLT16.TED.tst2013',
              'test': 'IWSLT16.TED.tst2014'}

    def __init__(self, src_lang, tgt_lang, mode='train', device=None):
        """ constructor
            :param src_lang: The src language
            :param tgt_lang: The target language
            :param mode: The train, valid, test mode
            :param device: A torch.device or str
        """
        assert mode in ['train', 'test', 'valid'], "Invalid mode {}".format(mode)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.mode = mode
        self.src_vocab = Vocab(lang=src_lang)
        self.tgt_vocab = Vocab(lang=tgt_lang)
        self.bpe_corpus_dir = os.path.join(ROOT_BPE_DIR, 'iwslt',
                                           '{}-{}'.format(src_lang[1:], tgt_lang[1:]))
        self.src_text, self.tgt_text = self.get_txt_name(mode)
        assert len(self.src_text) == len(self.tgt_text), "Not paired datset, something is wrong"
        if type(device) == str:
            device = torch.device(device)
        self.device = device

    def get_txt_name(self, mode='train'):
        """ Return the name of the txt """
        base_text_name = '{}.' + self.src_lang[1:] + '-' \
                         + self.tgt_lang[1:]
        base_text_name_src = base_text_name + self.src_lang
        base_text_name_tgt = base_text_name + self.tgt_lang
        src_txt = os.path.join(self.bpe_corpus_dir,
                               base_text_name_src.format(self.prefix[mode]))
        tgt_txt = os.path.join(self.bpe_corpus_dir,
                               base_text_name_tgt.format(self.prefix[mode]))
        with open(src_txt, 'r') as src_file, open(tgt_txt, 'r') as tgt_file:
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

class IWSLTExample(namedtuple('Example', ('src', 'src_lengths', 'tgt', 'tgt_lengths'))):
    """ A data structure """
    pass

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
        self.device = self.dataset.device
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
                            device=self.device,
                            dtype=torch.int64)
