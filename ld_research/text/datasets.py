""" Contains object for dataset and vocab
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from ld_research.settings import EN, ROOT_BPE_DIR, FR, BOS_WORD, EOS_WORD, LOGGER, UNK_WORD, PAD_WORD
from ld_research.text.utils import get_vocab_file
import os

class Vocab:
    """ the vocab object """
    def __init__(self, lang):
        """ constructor """
        self.vocab_file = get_vocab_file(lang)
        self.words_with_freq = dict()
        with open(self.vocab_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word, freq = line.split()
                self.words_with_freq[word] = freq

        # Add utlity tokens
        self.words_with_freq[BOS_WORD] = 1
        self.words_with_freq[EOS_WORD] = 1
        self.words_with_freq[UNK_WORD] = 1
        self.words_with_freq[PAD_WORD] = 1

        # get words
        self.idx2words = list(self.words_with_freq.keys())
        self.words2idx = {word: self.idx2words.index(word)
                          for word in self.idx2words}

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

    def numerize(self, sentence):
        """ return a list of int """
        return [self.get_index(word) for word in sentence.split()]

    def denumerize(self, indexs):
        """ return a list of char from indexes """
        return ' '.join([self.get_word(index) for index in indexs])

class IWSLTExample(namedtuple('example', ('src', 'tgt'))):
    """ Holding result """
    pass

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
        src_sentence = self.src_text[index]
        tgt_sentence = self.tgt_text[index]
        src_ids = self.src_vocab.numerize(src_sentence)
        tgt_ids = self.tgt_vocab.numerize(tgt_sentence)

        # Append EOS at the end
        src_ids.append(self.src_vocab.get_index(EOS_WORD))
        tgt_ids.append(self.tgt_vocab.get_index(EOS_WORD))

        # Append BOS to tgt
        tgt_ids.insert(0, self.tgt_vocab.get_index(BOS_WORD))
        return IWSLTExample(src=src_ids, tgt=tgt_ids)

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

    def _pad_to_same_length(self, sentences, pad_index):
        """ Given a list of sentences. Pad each to the maximum length.
            :param sentences: A list of list of indices
            :param pad_index: The index standing for <PAD>
            :return A list of list with same length
        """
        max_len = max([len(sentence) for sentence in sentences])
        new_sentences = []
        for sentence in sentences:
            new_sentence = sentence[:]
            new_sentence += [pad_index] * (max_len - len(new_sentence))
            new_sentences += [new_sentence]
        return new_sentences

    def collate_fn(self, batch):
        """ Merge a list of IWSLTExample into one IWSLTExample """
        new_src = self._pad_to_same_length(sentences=[example.src for example in batch],
                                           pad_index=self.dataset.src_vocab.get_index(PAD_WORD))
        new_tgt = self._pad_to_same_length(sentences=[example.tgt for example in batch],
                                           pad_index=self.dataset.tgt_vocab.get_index(PAD_WORD))
        return IWSLTExample(src=torch.tensor(new_src, device=self.device),
                            tgt=torch.tensor(new_tgt, device=self.device))

if __name__ == '__main__':
    dset = IWSLTDataset(src_lang=FR, tgt_lang=EN, mode='test')
    test_loader = IWSLTDataloader(dataset=dset, batch_size=10)
    examples = next(iter(test_loader))



