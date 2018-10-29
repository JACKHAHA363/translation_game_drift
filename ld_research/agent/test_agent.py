""" Test GRU
"""
import torch
from ld_research.agent.grus import GRUDecoder, GRUEncoder

VOCAB_SIZE = 500
EMB_SIZE = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 4
SEQ_LEN = 6
TGT_SEQ_LEN = 5

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
    tgt = _prepare_batch(TGT_SEQ_LEN)
    states, memory = encoder.encode(src=src)

    # Teacher force
    samples = decoder.greedy_decoding(bos_id=0, eos_id=VOCAB_SIZE-1,
                                      memory=memory, states=states,
                                      max_steps=20)
    samples = samples[:, 1:]
    assert samples.size(0) == BATCH_SIZE
    if samples.size(1) < 20:
        endings = samples[:, -1]
        assert (endings == VOCAB_SIZE - 1).sum().item() == BATCH_SIZE
    elif samples.size(1) > 20:
        assert False

def test_agent():
    """ Test and agent """
    pass

if __name__ == '__main__':
    test_agent()