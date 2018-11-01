"""
    Created by yuchen on 10/28/18
    Description:
"""
from ld_research.settings import EN, FR, DE
from ld_research.text import IWSLTDataset, IWSLTDataloader, Multi30KLoader, Multi30KDataset

def test_iwslt():
    """ Test iwslt data loader """
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
