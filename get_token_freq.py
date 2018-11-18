""" Plot the token freq of a FR-EN agent
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from ld_research.settings import FR, EN, BOS_WORD, EOS_WORD, UNK_WORD
from ld_research.text import Vocab, IWSLTDataset, IWSLTDataloader
from ld_research.model import Agent
from ld_research.model.utils import sequence_mask


def get_args():
    """ Get args """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_path', required=True, type=str,
                        help='path to checkpoint')
    parser.add_argument('-adjust', action='store_true',
                        help='adjust the utility token')
    return parser.parse_args()


if __name__ == '__main__':
    # Main loop
    args = get_args()
    ckpt = torch.load(args.ckpt_path,
                      map_location=lambda storage, loc: storage)
    model_state_dict = ckpt['agent']
    fr_vocab = Vocab(FR)
    en_vocab = Vocab(EN)
    opt = argparse.Namespace
    opt.emb_size = 256
    opt.hidden_size = 256
    opt.dropout = 0.1
    agent = Agent(fr_vocab, en_vocab, opt)
    agent.load_state_dict(model_state_dict)

    # Test dataset
    dataset = IWSLTDataset(FR, EN, mode='valid')
    dataloader = IWSLTDataloader(dataset, batch_size=200, shuffle=True)

    # Get token freq
    # Init to be -reference
    freq_diff_np = np.zeros(len(en_vocab))

    # For all data loader
    for batch in dataloader:
        trans_en, trans_en_lengths = agent.batch_translate(src=batch.src,
                                                           src_lengths=batch.src_lengths,
                                                           max_lengths=batch.tgt_lengths,
                                                           method='greedy')
        en, en_lengths = batch.tgt, batch.tgt_lengths

        # Increment my words
        model_words = trans_en.masked_select(mask=sequence_mask(trans_en_lengths,
                                                                max_len=trans_en.size(1)))
        for word in model_words.tolist():
            freq_diff_np[word] -= 1

        # Minus reference
        ref_words = en.masked_select(mask=sequence_mask(en_lengths,
                                                        max_len=en.size(1)))
        for word in ref_words.tolist():
            freq_diff_np[word] += 1

    # Don't count BOS, EOS, UNK
    if args.adjust:
        freq_diff_np[en_vocab.get_index(BOS_WORD)] = 0
        freq_diff_np[en_vocab.get_index(EOS_WORD)] = 0
        freq_diff_np[en_vocab.get_index(UNK_WORD)] = 0

    # Plot this
    plt.plot(np.log(np.arange(0, len(en_vocab))), freq_diff_np / 1000)
    plt.show()










