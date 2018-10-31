""" Contains main method for GRU
"""
import torch
from ld_research.model.utils import GlobalAttention

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
        self.dropout = 0.3
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

    def greedy_decoding(self, bos_id, eos_id, memory, states, max_steps=None, memory_lengths=None):
        """ Perform greedy decoding
            :param bos_id: The initial id to start sample
            :param eos_id: The id for ending
            :param memory: [bsz, src_seq_len, memory_size]
            :param states: encoder states. [1, bsz, hidden_size]
            :param max_steps: maximum decoding length. excluding BOS_WORD. None if no maximum
            :param memory_lengths: [bsz]
            :return sample_ids: [bsz, sample_len]
        """
        bsz = memory.size(0)
        device = memory.device
        init_inputs = torch.zeros(size=[bsz, 1], dtype=torch.int64,
                                  device=device)
        init_inputs.fill_(bos_id)

        def finished(inputs, time_step):
            """ Given a batch of current ids and time step, determine if it's finished """
            if max_steps and time_step == max_steps:
                return True
            all_ends = torch.sum(inputs == eos_id).item()
            return all_ends == bsz

        def sample(logits):
            """ sample the ids from logits [bsz, vocab_size] """
            _, ids = torch.max(logits, dim=-1)
            return ids.unsqueeze(1)

        # Ready to sample
        results =[init_inputs]
        step = 0
        curr_inputs = init_inputs
        curr_hidden = states
        while not finished(curr_inputs, step):
            # embedding [bsz, 1, inp_size]
            curr_emb = self.embeddings(curr_inputs)

            # [bsz, 1, hidden_size]
            decoder_outputs, next_hidden = self.gru(curr_emb,
                                                    curr_hidden)
            context, _ = self.attn(query=decoder_outputs,
                                   memory_bank=memory,
                                   memory_lengths=memory_lengths)
            context = self.dropout(context)

            # Get logits
            # [bsz, hidden_size]
            context = context.squeeze(1)
            logits = self.linear_out(context)
            sampled_ids = sample(logits)

            # Update curr_inputs and step
            step += 1
            curr_inputs = sampled_ids
            curr_hidden = next_hidden

            # Appending resutls
            results.append(sampled_ids)
        return torch.cat(results, dim=1)
