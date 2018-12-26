""" Take a checkpoint and translate the sentence to EN to check drifting
"""
import torch
import argparse
import os

from ld_research.text import Multi30KLoader, Multi30KDataset
from ld_research.text import Vocab
from ld_research.settings import FR, EN
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
    parser.add_argument('-out', required=True,
                        help='Path to output file.')
    return parser.parse_args()


if __name__ == '__main__':
    """ main loop """
    args = get_args()
    agents = {os.path.basename(ckpt_path): get_fr_en_agent(ckpt_path) for ckpt_path in args.ckpt}
    for name in agents:
        agents[name].eval()
    dataloader = Multi30KLoader(Multi30KDataset('valid'), 200, False)
    with open(args.out, 'w') as f:
        for batch in dataloader:
            # Translate
            batch_result = {name: agents[name].batch_translate(src=batch.fr,
                                                               src_lengths=batch.fr_lengths,
                                                               max_lengths=batch.fr_lengths,
                                                               method='greedy')[0]
                            for name in agents}

            for batch_idx in range(batch.fr.size(0)):
                line = "french: {:>10}\nReference: {:>10}\n".format(
                    FR_VOCAB.to_readable_sentences(batch.fr[batch_idx])[0],
                    EN_VOCAB.to_readable_sentences(batch.en[batch_idx])[0])
                for name in batch_result:
                    batch_sent = EN_VOCAB.to_readable_sentences(batch_result[name][batch_idx])[0]
                    line += "{}: {:>10}\n".format(name, batch_sent)
                line += '\n'
                f.write(line)
