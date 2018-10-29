""" The general interface of an agent
"""
import torch

class GRUEncoder(torch.nn.Module):
    """ The GRU Encoder """
    def __init__(self, embedding, num_layers=1, hidden_size=256,
                 dropout=0.3):
        """ Constructor
            :param embedding: An instance of torch.nn.Embedding
            :param num_layers: num of layers in GRU
            :param hidden_size: GRU size
            :param dropout: dropout value
        """
        super(GRUEncoder, self).__init__()
        assert isinstance(embedding, torch.nn.Embedding)
        self.embedding = embedding
        self.input_size = embedding.embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = 0.3
        self.gru = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout)


    def forward(self, src, lengths=None):
        """ Forward encoder
            :param src: LongTensor. [seq_len, bsz, input_size]
            :param lengths: LongTensor. Lengths of each seq. [bsz]
            :return: (states, memory).
                states: Final encoder state.
                memory: The memory bank for attention. `[seq_len, bsz, num_hidden]`
        """


class Agent:
    """ The agent """
    pass

if __name__ == '__main__':
    torch.nn.utils.rnn
