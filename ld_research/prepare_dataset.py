""" prepare the ISWIT and Multi30k task1 dataset.
"""
from ld_research.settings import FR, EN, DE
from ld_research.text.utils import prepare_IWSLT, prepare_multi30k, learn_bpe, apply_bpe_iwslt, apply_bpe_multi30k

if __name__ == '__main__':
    """ Main logic """
    # Get corpus and tokenize
    prepare_IWSLT()
    prepare_multi30k()

    # Learn the BPE
    learn_bpe()

    # Apply BPE to IWSLT
    apply_bpe_iwslt(FR, EN)
    apply_bpe_iwslt(EN, DE)

    # Apply BPE to Multi30k
    apply_bpe_multi30k()
