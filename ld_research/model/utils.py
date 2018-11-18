""" Contains some utilities
"""
from torch import nn
import torch
from torch.nn import functional as F


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GlobalAttention(nn.Module):
    """ Adapted from openNMT-py
    """
    def __init__(self, dim, attn_type="general"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

    def score(self, query, memory):
        """
        Args:
          query (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          memory (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """

        # Check input sizes
        src_batch, src_len, src_dim = memory.size()
        tgt_batch, tgt_len, tgt_dim = query.size()
        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = query.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                query = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = memory.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(query, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(query.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(memory.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)
            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, query, memory_bank, memory_lengths=None):
        """

        Args:
          query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): query vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the query context lengths `[batch]`
        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[batch x tgt_len x dim]`
          * Attention distribtutions for each query
             `[batch x tgt_len x src_len]`
        """

        # one step input
        if query.dim() == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = query.size()

        # compute attention scores, as in Luong et al.
        align = self.score(query.contiguous(), memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the query hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, query], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        return attn_h, align_vectors
