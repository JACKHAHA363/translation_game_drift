"""
    Created by yuchen on 10/28/18
    Description:
"""
from ld_research.settings import EN, FR
from ld_research.text import IWSLTDataset, IWSLTDataloader

def test_iwslt():
    """ Test data loader """
    dset = IWSLTDataset(src_lang=FR, tgt_lang=EN, mode='test')
    test_loader = IWSLTDataloader(dataset=dset, batch_size=10)
    examples = next(iter(test_loader))
    src, src_lengths, tgt, tgt_lengths = examples.src, \
                                         examples.src_lengths, \
                                         examples.tgt, \
                                         examples.tgt_lengths
    assert src.size(0) == 10
    assert src_lengths.size(0) == 10
    assert tgt.size(0) == 10
    assert tgt_lengths.size(0) == 10
