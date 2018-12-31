""" Take a checkpoint and translate the sentence to EN to check drifting
"""
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from ld_research.text import Multi30KLoader, Multi30KDataset, IWSLTDataset, IWSLTDataloader
from ld_research.text import Vocab
from ld_research.settings import FR, EN, EOS_WORD, BOS_WORD, UNK_WORD
from ld_research.training.finetune import AgentType
from ld_research.model import Agent

FR_VOCAB = Vocab(FR)
EN_VOCAB = Vocab(EN)


def get_fr_en_agent(ckpt_path):
    """ Return a fr_en_agent """
    opt = argparse.Namespace
    opt.emb_size = 256
    opt.hidden_size = 256
    opt.dropout = 0.1
    agent = Agent(FR_VOCAB, EN_VOCAB, opt)

    # Get state dict
    ckpt = torch.load(ckpt_path,
                      map_location=lambda storage, loc: storage)
    model_state_dict = ckpt.get('agent')
    if model_state_dict is None:
        model_state_dict = ckpt.get(AgentType.FR_EN).get('agent_state_dict')
    if model_state_dict is None:
        raise ValueError('Something wrong with checkpoint')

    agent.load_state_dict(model_state_dict)
    return agent


def get_args():
    """ parse from cmd """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', nargs='+', required=True,
                        help='Take in a (list of) of checkpoint path.')
    parser.add_argument('-data', choices=['iwslt', 'multi30k'], default='multi30k',
                        help='The name of corpus to do token frequency')
    parser.add_argument('-out', required=True,
                        help='Path to output translation file.')
    return parser.parse_args()


def accumulate_tok_stats(tok_freq, sent):
    """ Tok_freq: A dictonary of {word: count}. sent a string of sentence """
    for word in sent.split():
        tok_freq[word] += 1
    return tok_freq


def to_sorted_np(dict):
    """ Sorted the values of `dict` and return a np array """
    result = np.array(list(dict.values()))
    result[::-1].sort()
    return result

def get_name(path):
    """ Get the name without ckpt """
    name = os.path.basename(path)
    name = name.split(sep='.')[0]
    return name

if __name__ == '__main__':
    """ main loop """
    args = get_args()
    agents = {get_name(ckpt_path): get_fr_en_agent(ckpt_path) for ckpt_path in args.ckpt}
    for name in agents:
        agents[name].eval()
    dataloader = Multi30KLoader(Multi30KDataset('valid'), 200, True) if args.data == 'multi30k' else \
        IWSLTDataloader(IWSLTDataset(src_lang=FR, tgt_lang=EN, mode='valid'), 200, True)

    # Determine Longest name for formatting
    longest_name_lengths = max([len(name) for name in agents])
    fmt_output = "{:<%d}\t{}\n" % longest_name_lengths

    # Tok freq analysis
    tok_freqs = {name: {word: 0 for word in EN_VOCAB.idx2words} for name in agents}
    ref_tok_freq = {word: 0 for word in EN_VOCAB.idx2words}

    # BLEU Score
    hyps = []
    refs = {name: [] for name in agents}
    with open(args.out, 'w') as f:
        for batch in dataloader:
            if args.data == 'iwslt':
                fr = batch.src
                fr_lengths = batch.src_lengths
                en = batch.tgt
            else:
                fr = batch.fr
                fr_lengths = batch.fr_lengths
                en = batch.en

            # Translate
            batch_result = {name: agents[name].batch_translate(src=fr,
                                                               src_lengths=fr_lengths,
                                                               max_lengths=1000,
                                                               method='greedy')[0]
                            for name in agents}
            for batch_idx in range(fr.size(0)):
                lines = ""
                fr_sent = FR_VOCAB.to_readable_sentences(fr[batch_idx])[0]
                lines += fmt_output.format("Fr", fr_sent)
                en_sent = EN_VOCAB.to_readable_sentences(en[batch_idx])[0]
                lines += fmt_output.format("Ref", en_sent)
                ref_tok_freq = accumulate_tok_stats(ref_tok_freq, en_sent)
                for name in batch_result:
                    sent = EN_VOCAB.to_readable_sentences(batch_result[name][batch_idx])[0]
                    lines += fmt_output.format(name, sent)
                    tok_freqs[name] = accumulate_tok_stats(tok_freqs[name], sent)
                lines += '\n'
                f.write(lines)

            # Collect for BLEU
            hyps += EN_VOCAB.to_sentences(en)
            for name in agents:
                refs[name] += [[sent] for sent in EN_VOCAB.to_sentences(batch_result[name])]

    # Adjust the frequency of funtional word
    for w in [EOS_WORD, BOS_WORD, UNK_WORD]:
        ref_tok_freq[w] = 0
        for name in tok_freqs:
            tok_freqs[name][w] = 0

    # Get token freq - ref token freq after sorting
    sorted_ref_tf = to_sorted_np(ref_tok_freq)
    diff_tok_freq = {}
    for name in tok_freqs:
        diff_tok_freq[name] = to_sorted_np(tok_freqs[name]) - sorted_ref_tf

    # Plotting
    plots = {name: plt.plot(np.log(np.arange(1, len(EN_VOCAB) + 1)),
                            diff_tok_freq[name] / 1000.)[0] for name in diff_tok_freq}
    plt.legend(list(plots.values()), list(plots.keys()))
    plt.xlabel('Vocab (log)')
    plt.ylabel('Freq Diff (k)')
    plt.title('Token Frequency - Reference ({})'.format(args.data))
    plt.savefig('token_freq_{}.png'.format(args.data))

    # Showing BLEU score
    for name in agents:
        print('{} BLEU: {:.2f}'.format(name,
                                       100 * corpus_bleu(list_of_references=refs[name],
                                                         hypotheses=hyps)))

    # Unique Token / Total Token
    for name, tok_freq in tok_freqs.items():
        nb_uniq_tok = len([word for word in tok_freq if tok_freq[word] > 0])
        total_tok = sum(tok_freq.values())
        print('Unique Token Percentage {}: {:.2f}'.format(name, nb_uniq_tok / float(total_tok) * 100))
