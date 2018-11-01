""" Data preprocessing loading utility
"""
from os.path import join

from ld_research.settings import LOGGER, ROOT_BPE_DIR, \
    BOS_WORD, EOS_WORD, UNK_WORD, PAD_WORD


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
            :param ids: list of list of word index (int) or list of ids
            :param excludes: List of words to exclude. If None default to BOS, EOS, PAD
            :return: sentences. A list of list of str.
        """
        excludes = excludes if excludes else [BOS_WORD, EOS_WORD, PAD_WORD]
        if type(ids[0]) is int:
            ids = [ids]
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
