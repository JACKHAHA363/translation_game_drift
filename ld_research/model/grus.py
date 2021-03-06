""" Contains main method for GRU
"""
import torch
from torch.distributions import Categorical
from ld_research.model.utils import GlobalAttention


def greedy_sample(logits):
    """ Greedy sample """
    _, ids = torch.max(logits, dim=-1)
    return ids


def random_sample(logits):
    """ sample the ids from logits [bsz, vocab_size].
        :return ids: [bsz]
    """
    dists = Categorical(logits=logits)
    return dists.sample()


SAMPLING = {'greedy': greedy_sample,
            'random': random_sample}


class GRUEncoder(torch.nn.Module):
    """ The GRU Encoder """
    def __init__(self, embeddings, num_layers=1, hidden_size=256,
                 dropout=0.3):
        """ Constructor
            :param embeddings: An instance of torch.nn.Embedding
            :param num_layers: num of layers in GRU
            :param hidden_size: GRU size
            :param dropout: dropout value
        """
        super(GRUEncoder, self).__init__()
        assert isinstance(embeddings, torch.nn.Embedding)
        self.embeddings = embeddings
        self.input_size = embeddings.embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True)

    def encode(self, src):
        """ Forward encoder
            :param src: LongTensor. [bsz, seq_len, input_size]
            :return: (states, memory).
                states: Final encoder state. [1, bsz, num_hidden]
                memory: The memory bank for attention. `[bsz, seq_len, num_hidden]`
        """
        emb = self.embeddings(src)
        memory, states = self.gru(emb)
        return states, memory


class GRUDecoder(torch.nn.Module):
    """ The decoder """
    def __init__(self, embeddings, num_layers=1, hidden_size=256,
                 dropout=0.3):
        """ Constructor """
        super(GRUDecoder, self).__init__()
        assert isinstance(embeddings, torch.nn.Embedding)
        self.embeddings = embeddings
        self.input_size = embeddings.embedding_dim
        self.output_size = embeddings.embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True)
        self.attn = GlobalAttention(dim=hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size,
                                          embeddings.num_embeddings,
                                          bias=True)

    def teacher_forcing(self, targets, memory, states, memory_lengths=None):
        """ Do a teacher forcing
            :param targets: [bsz, seq_len]
            :param memory: [bsz, src_seq_len, memory_size]
            :param states: encoder states. [1, bsz, hidden_size]
            :param memory_lengths: [bsz]
            :return: (logits, alignments)
                logits: [bsz, seq_len, vocab_size]
                alignments: [bsz, seq_len, src_seq_len]
        """
        # decoder_outputs
        # [bsz, seq_len, hidden_size]
        bsz = targets.size(0)
        seq_len = targets.size(1)
        embs = self.embeddings(targets)
        decoder_outputs, _ = self.gru(embs, states)

        # Compute attention
        context, alignments = self.attn(query=decoder_outputs,
                                        memory_bank=memory,
                                        memory_lengths=memory_lengths)
        context = self.dropout(context)

        # Compute logits
        logits = self.linear_out(context.view(-1, self.hidden_size))
        logits = logits.view(bsz, seq_len, self.embeddings.num_embeddings)
        return logits, alignments

    def decode(self, bos_id, eos_id, memory, states,
               max_steps=None, memory_lengths=None, method='greedy'):
        """ Perform greedy decoding for max_steps
            :param bos_id: The initial id to start sample
            :param eos_id: The id for ending
            :param memory: [bsz, src_seq_len, memory_size]
            :param states: encoder states. [1, bsz, hidden_size]
            :param max_steps: None or int or tensor of [bsz]. maximum decoding length. Excluding BOS_WORD and EOS_WORD
            :param memory_lengths: [bsz]
            :param method: one of 'greedy' or 'random'
            :return (sample_ids, lengths):
                sample_ids: [bsz, sample_lengths] The sampled_ids.
                sample_lengths: [bsz] The length of each sentence including BOS and EOS
        """
        bsz = memory.size(0)
        device = memory.device
        init_inputs = torch.zeros(size=[bsz], dtype=torch.int64,
                                  device=device)
        init_inputs.fill_(bos_id)

        def get_finished(last_ids, lengths, finished):
            """ Determine whether finish decoding
                :param last_ids: A tensor of [bsz] with last sampled id (Already appended)
                :param lengths: A tensor of [bsz] indicating lengths
                :param finished: A tensor of [bsz] indicating if finished
                return: finished: A tensor of [bsz] indicating whether or not it's finished
            """
            # Those who sample eos_id is marked finished
            new_finished = (last_ids == eos_id).int()

            # Or exceeding max steps
            if max_steps is not None:
                new_finished += (lengths >= max_steps + 1).int()
            new_finished += finished
            new_finished = (new_finished > 0).int()
            return new_finished

        # Get sample mechanism
        get_ids = SAMPLING.get(method)
        if get_ids is None:
            raise ValueError('sample method {} not implemented'.format(method))

        # Ready to sample
        results =[init_inputs]
        lengths = torch.ones(bsz).to(device=device).int()   # already ones since BOS
        finished = torch.zeros(bsz).to(device=device).int()
        last_ids = init_inputs
        last_hidden = states
        while True:
            # See if finished
            finished = get_finished(last_ids=last_ids,
                                    lengths=lengths,
                                    finished=finished)
            if torch.sum(finished).item() == bsz:
                break

            # embedding [bsz, 1, inp_size]
            emb = self.embeddings(last_ids.unsqueeze(1))

            # [bsz, 1, hidden_size]
            decoder_outputs, hidden = self.gru(emb,
                                               last_hidden)
            context, _ = self.attn(query=decoder_outputs,
                                   memory_bank=memory,
                                   memory_lengths=memory_lengths)
            context = self.dropout(context)

            # Get logits
            # [bsz, hidden_size]
            context = context.squeeze(1)
            logits = self.linear_out(context)

            # [bsz]
            sampled_ids = get_ids(logits)
            sampled_ids = sampled_ids * (1 - finished.long())
            sampled_ids += eos_id * finished.long()

            # Update curr_inputs and lengths
            last_eos = results[-1] == eos_id
            lengths += (1 - last_eos.int())
            results.append(sampled_ids)

            # Update next input
            last_ids = sampled_ids
            last_hidden = hidden

        # Update final lengths with EOS
        final_ids = (torch.ones([bsz]) * eos_id).long().to(device=device)
        last_eos = results[-1] == eos_id
        lengths += (1 - last_eos.int())
        results.append(final_ids)
        return torch.stack(results, dim=1), lengths


class GRUValueDecoder(torch.nn.Module):
    """ The Value decoder """
    def __init__(self, embeddings, num_layers=1, hidden_size=256,
                 dropout=0.3):
        """ Constructor """
        super(GRUValueDecoder, self).__init__()
        assert isinstance(embeddings, torch.nn.Embedding)
        self.embeddings = embeddings
        self.input_size = embeddings.embedding_dim
        self.output_size = embeddings.embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True)
        self.attn = GlobalAttention(dim=hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size,
                                          1,
                                          bias=True)

    def forward(self, targets, memory, states, memory_lengths=None):
        """ Do a teacher forcing
            :param targets: [bsz, seq_len]
            :param memory: [bsz, src_seq_len, memory_size]
            :param states: encoder states. [1, bsz, hidden_size]
            :param memory_lengths: [bsz]
            :return: values [bsz, seq_len]
        """
        # decoder_outputs
        # [bsz, seq_len, hidden_size]
        bsz = targets.size(0)
        seq_len = targets.size(1)
        embs = self.dropout(self.embeddings(targets))
        decoder_outputs, _ = self.gru(embs, states)

        # Compute attention
        context, _ = self.attn(query=decoder_outputs,
                               memory_bank=memory,
                               memory_lengths=memory_lengths)
        context = self.dropout(context)

        # Compute values
        values = self.linear_out(context.view(-1, self.hidden_size))
        values = values.view(bsz, seq_len, 1).squeeze(-1)
        return values
