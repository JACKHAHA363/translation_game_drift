""" The general interface of an agent
"""
import torch
from ld_research.agent.grus import GRUDecoder, GRUEncoder

class Agent:
    """ The agent. A seq to seq model """
    def __init__(self, src_vocab, tgt_vocab):
        """ Constructor
            :param src_vocab: source language_vocab. An instance of `Vocab`
            :param tgt_vocab: target language_vocab. An instance of `Vocab`
        """
        self.src_emb = torch.nn.Embedding()


if __name__ == '__main__':
    print('hello')
