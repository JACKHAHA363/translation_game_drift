"""
    Created by yuchen on 10/28/18
    Description:
"""
import torch

from ld_research.settings import EN, FR, DE, BOS_WORD, EOS_WORD
from ld_research.text import IWSLTDataset, IWSLTDataloader, Multi30KLoader, Multi30KDataset

def test_iwslt():
    """ Test iwslt data loader """
    dset = IWSLTDataset(src_lang=FR, tgt_lang=EN, mode='test')
    test_loader = IWSLTDataloader(dataset=dset, batch_size=10)
    src_vocab = test_loader.src_vocab
    tgt_vocab = test_loader.tgt_vocab
    examples = next(iter(test_loader))
    src, src_lengths, tgt, tgt_lengths = examples.src, \
                                         examples.src_lengths, \
                                         examples.tgt, \
                                         examples.tgt_lengths
    assert src.size(0) == 10
    assert src_lengths.size(0) == 10
    assert tgt.size(0) == 10
    assert tgt_lengths.size(0) == 10
    for src_ids, src_length, tgt_ids, tgt_length in zip(src, src_lengths, tgt, tgt_lengths):
        # Verify src
        src_length = src_length.item()
        src_eos = src_vocab.get_index(EOS_WORD)
        assert verify_eos(eos_id=src_eos, ids=src_ids, length=src_length)

        # Verify tgt
        tgt_length = tgt_length.item()
        tgt_eos = tgt_vocab.get_index(EOS_WORD)
        tgt_bos = tgt_vocab.get_index(BOS_WORD)
        assert tgt_ids[0].item() == tgt_bos
        assert verify_eos(eos_id=tgt_eos, ids=tgt_ids[1:], length=tgt_length - 1)

def test_multi30k():
    """ Test multi30k data loader """
    dset = Multi30KDataset(mode='test')
    test_loader = Multi30KLoader(dataset=dset, batch_size=10)
    examples = next(iter(test_loader))
    _, _, _, _, _, _ = examples.fr, \
                       examples.fr_lengths, \
                       examples.en, \
                       examples.en_lengths, \
                       examples.de, \
                       examples.de_lengths
    for lang in [FR, EN, DE]:
        assert examples.id_dicts[lang].size(0) == 10
        assert examples.length_dicts[lang].size(0) == 10


def verify_eos(eos_id, ids, length):
    """ Return a bool.
        :param eos_id: The id for EOS_WORD
        :param ids: A tensor of [length]
        :param length: An int
    """
    if ids[length - 1].item() != eos_id:
        return False
    if torch.sum((ids[:length - 1] == eos_id).int()).item() != 0:
        return False
    remain_src = ids[length:]
    if len(remain_src) > 0:
        if torch.sum((remain_src == eos_id).int()).item() != len(remain_src):
            return False
    return True
