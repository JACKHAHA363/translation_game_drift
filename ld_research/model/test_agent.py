""" Test GRU
"""
import torch
import argparse
from ld_research.model.grus import GRUDecoder, GRUEncoder
from ld_research.model import Agent, ValueNetwork
from ld_research.settings import FR, EN
from ld_research.text import Vocab, IWSLTDataloader, IWSLTDataset

# Some config
VOCAB_SIZE = 500
EMB_SIZE = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 200
SEQ_LEN = 6
TGT_SEQ_LEN = 5
SRC_LANG = FR
TGT_LANG = EN

def _prepare_batch(seq_len):
    """ Get a batch of fake data """
    fakedata = torch.randint(high=VOCAB_SIZE, size=[BATCH_SIZE, seq_len],
                             dtype=torch.int64)
    return fakedata

def test_encoder():
    """ Test encoder """
    embeddings = torch.nn.Embedding(num_embeddings=VOCAB_SIZE,
                                    embedding_dim=EMB_SIZE)
    encoder = GRUEncoder(embeddings, hidden_size=HIDDEN_SIZE)
    batch = _prepare_batch(SEQ_LEN)
    states, memory_bank = encoder.encode(src=batch)
    assert states.size(0) == 1
    assert states.size(1) == BATCH_SIZE
    assert states.size(2) == HIDDEN_SIZE
    assert memory_bank.size(0) == BATCH_SIZE
    assert memory_bank.size(1) == SEQ_LEN
    assert memory_bank.size(2) == HIDDEN_SIZE

def test_decoder_tf():
    """ Test teacher forcing """
    embeddings = torch.nn.Embedding(num_embeddings=VOCAB_SIZE,
                                    embedding_dim=EMB_SIZE)
    encoder = GRUEncoder(embeddings, hidden_size=HIDDEN_SIZE)
    decoder = GRUDecoder(embeddings, hidden_size=HIDDEN_SIZE)
    src = _prepare_batch(SEQ_LEN)
    tgt = _prepare_batch(TGT_SEQ_LEN)
    states, memory = encoder.encode(src=src)

    # Teacher force
    logits, alignments = decoder.teacher_forcing(tgt, memory=memory, states=states)
    assert logits.size(0) == BATCH_SIZE
    assert logits.size(1) == TGT_SEQ_LEN
    assert logits.size(2) == VOCAB_SIZE
    assert alignments.size(0) == BATCH_SIZE
    assert alignments.size(1) == TGT_SEQ_LEN
    assert alignments.size(2) == SEQ_LEN

def test_decoder_greedy():
    """ Test greedy """
    embeddings = torch.nn.Embedding(num_embeddings=VOCAB_SIZE,
                                    embedding_dim=EMB_SIZE)
    encoder = GRUEncoder(embeddings, hidden_size=HIDDEN_SIZE)
    decoder = GRUDecoder(embeddings, hidden_size=HIDDEN_SIZE)
    src = _prepare_batch(SEQ_LEN)
    states, memory = encoder.encode(src=src)

    # Greedy decode
    bos_id = 0
    eos_id = VOCAB_SIZE - 1
    max_steps = 50
    samples, sample_lengths = decoder.greedy_decoding(bos_id=bos_id,
                                                      eos_id=eos_id,
                                                      memory=memory,
                                                      states=states,
                                                      max_steps=max_steps)
    assert samples.size(0) == BATCH_SIZE
    assert samples.size(1) <= max_steps + 2
    for sample, length in zip(samples, sample_lengths):
        length = length.item()
        assert length <= max_steps + 2

        # BOS at beginning
        assert sample[0].item() == bos_id

        # EOS at length -1
        assert sample[length - 1].item() == eos_id

        # No EOS in between
        between_eos = (sample[1:length - 2] == eos_id).int()
        assert torch.sum(between_eos).item() == 0

        # Rest is eos
        rest = sample[length:]
        if len(rest) > 0:
            assert torch.sum((rest == eos_id).int()).item() == len(rest)


def test_decoder_greedy_tensor():
    """ Test greedy """
    embeddings = torch.nn.Embedding(num_embeddings=VOCAB_SIZE,
                                    embedding_dim=EMB_SIZE)
    encoder = GRUEncoder(embeddings, hidden_size=HIDDEN_SIZE)
    decoder = GRUDecoder(embeddings, hidden_size=HIDDEN_SIZE)
    src = _prepare_batch(SEQ_LEN)
    states, memory = encoder.encode(src=src)

    # Greedy decode
    bos_id = 0
    eos_id = VOCAB_SIZE - 1
    max_steps = torch.randint(SEQ_LEN, size=[BATCH_SIZE]).int()
    samples, sample_lengths = decoder.greedy_decoding(bos_id=bos_id,
                                                      eos_id=eos_id,
                                                      memory=memory,
                                                      states=states,
                                                      max_steps=max_steps)
    assert samples.size(0) == BATCH_SIZE
    assert samples.size(1) <= torch.max(max_steps).item() + 2
    for sample, length, max_step in zip(samples, sample_lengths, max_steps):
        length = length.item()
        max_step = max_step.item()
        if length == 2:
            pass
        assert length <= max_step + 2

        # BOS at beginning
        assert sample[0].item() == bos_id

        # EOS at length -1
        assert sample[length - 1].item() == eos_id

        # No EOS in between
        between_eos = (sample[1:length - 2] == eos_id).int()
        assert torch.sum(between_eos).item() == 0

        # Rest is padding
        rest = sample[length:]
        if len(rest) > 0:
            assert torch.sum((rest == eos_id).int()).item() == len(rest)


def test_agent():
    """ test agent """
    src_vocab = Vocab('.fr')
    tgt_vocab = Vocab('.en')
    opt = argparse.Namespace
    opt.emb_size = 256
    opt.hidden_size = 256
    agent = Agent(src_vocab, tgt_vocab, opt)

    # Test dataset
    dataset = IWSLTDataset(mode='test', src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    dataloader = IWSLTDataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))

    # With alignment
    logprobs, targets, masks, aligns = agent.forward(batch.src,
                                                     batch.tgt,
                                                     batch.src_lengths,
                                                     batch.tgt_lengths, with_align=True)
    assert list(logprobs.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1, len(tgt_vocab)]
    assert list(targets.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1]
    assert list(masks.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1]
    assert list(aligns.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1, batch.src.size(1)]

def test_value_agent():
    """ test agent """
    src_vocab = Vocab('.fr')
    tgt_vocab = Vocab('.en')
    opt = argparse.Namespace
    opt.value_emb_size = 256
    opt.value_hidden_size = 256
    agent = ValueNetwork(src_vocab, tgt_vocab, opt)

    # Test dataset
    dataset = IWSLTDataset(mode='test', src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    dataloader = IWSLTDataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))

    # With alignment
    values, masks, aligns = agent.forward(batch.src,
                                          batch.tgt,
                                          batch.src_lengths,
                                          batch.tgt_lengths, with_align=True)
    assert list(values.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1]
    assert list(masks.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1]
    assert list(aligns.size()) == [BATCH_SIZE, batch.tgt.size(1) - 1, batch.src.size(1)]
