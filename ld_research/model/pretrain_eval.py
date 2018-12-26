""" Evaluate a checkpoint on a corpus for BLEU score
"""
import argparse
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from ld_research.parser import add_pretrain_args
from ld_research.text import Multi30KLoader, Multi30KDataset
from ld_research.text import IWSLTDataset, IWSLTDataloader
from ld_research.text.iwslt import IWSLTExample
from ld_research.text import Vocab
from ld_research.settings import FR, EN, DE
from ld_research.training.finetune import AgentType
from ld_research.model import Agent


def get_args():
    """ Return args """
    parser = argparse.ArgumentParser('Argument parser')
    parser.add_argument('-ckpt', required=True,
                        help='path to the checkpoint')
    parser.add_argument('-dataset', default='iwslt', type=str,
                        choices=['iwslt', 'multi30k'],
                        help='choose the dataset')
    parser = add_pretrain_args(parser)
    return parser.parse_args()


def get_agent(ckpt_path, src_vocab, tgt_vocab):
    """ Return a fr_en_agent """
    opt = argparse.Namespace
    opt.emb_size = 256
    opt.hidden_size = 256
    opt.dropout = 0.1
    agent = Agent(src_vocab, tgt_vocab, opt)

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


def get_dataloader(args):
    """ Return the corresponding dataloader """
    if args.dataset == 'iwslt':
        return IWSLTDataloader(IWSLTDataset(args.src_lang, args.tgt_lang, mode='valid'))
    elif args.dataset == 'multi30k':
        return Multi30KLoader(Multi30KDataset('valid'), 200, False)
    else:
        raise ValueError('Invalid dataset value: {}'.format(args.dataset))


def multi30k_to_iwslt(multi_batch, src_lang, tgt_lang):
    """ Transform a multi30k batch to a iwslt batch for given src and tgt """
    src = multi_batch.id_dicts[src_lang]
    tgt = multi_batch.id_dicts[tgt_lang]
    src_langths = multi_batch.length_dicts[src_lang]
    tgt_langths = multi_batch.length_dicts[tgt_lang]

    # Remove BOS in en if used as input
    if src_lang == EN:
        src = src[:, 1:]
        src_langths -= 1
    return IWSLTExample(src=src, src_lengths=src_langths,
                        tgt=tgt, tgt_lengths=tgt)


if __name__ == '__main__':
    """ Get args """
    args = get_args()
    src_vocab = Vocab(args.src_lang)
    tgt_vocab = Vocab(args.tgt_lang)
    dataloader = get_dataloader(args)

    # Get model
    agent = get_agent(args.ckpt, src_vocab, tgt_vocab)
    agent.eval()

    # Main loop
    hyps = []
    refs = []
    for batch in tqdm(dataloader):
        # Get info from batch
        if args.dataset == 'multi30k':
            batch = multi30k_to_iwslt(batch, args.src_lang, args.tgt_lang)

        preds, pred_lengths = agent.batch_translate(batch.src,
                                                    batch.src_lengths,
                                                    max_lengths=100,
                                                    method='greedy')
        hyps += tgt_vocab.to_sentences(preds)
        refs += [[tgt_sent] for tgt_sent in tgt_vocab.to_sentences(batch.tgt)]

    score = corpus_bleu(list_of_references=refs,
                        hypotheses=hyps, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method3) * 100
    print('BLEU Score: {}'.format(score))
