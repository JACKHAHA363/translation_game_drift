""" Pretrain the agent on IWSLT
"""
import argparse
from ld_research.settings import FR, EN, DE
from ld_research.training.iwslt import Trainer

if __name__ == '__main__':
    """ Parse from command """
    parser = argparse.ArgumentParser('Pretrain')
    parser.add_argument('-src_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    parser.add_argument('-tgt_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-label_smoothing', default=0.1, type=float)
    parser.add_argument('-device', default='cpu', type=str,
                        help='The device. None, cpu, cuda, cuda/cpu:rank')

    # optimizer and learning rate
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size for training')
    parser.add_argument('-accum_count', type=int, default=1,
                        help="""Accumulate gradient this many times.
                        Approximately equivalent to updating
                        batch_size * accum_count batches at once.
                        Recommended for Transformer.""")
    parser.add_argument('-valid_steps', type=int, default=10000,
                        help='Perfom validation every X steps')
    parser.add_argument('-train_steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('-optim', default='sgd',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-adagrad_accumulator_init', type=float, default=0,
                        help="""Initializes the accumulator values in adagrad.
                        Mirrors the initial_accumulator_value option
                        in the tensorflow adagrad (use 0.1 for their default).
                        """)
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    parser.add_argument('-adam_beta1', type=float, default=0.9,
                        help="""The beta1 parameter used by Adam.
                        Almost without exception a value of 0.9 is used in
                        the literature, seemingly giving good results,
                        so we would discourage changing this value from
                        the default without due consideration.""")
    parser.add_argument('-adam_beta2', type=float, default=0.999,
                        help="""The beta2 parameter used by Adam.
                        Typically a value of 0.999 is recommended, as this is
                        the value suggested by the original paper describing
                        Adam, and is also the value adopted in other frameworks
                        such as Tensorflow and Kerras, i.e. see:
                        https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                        https://keras.io/optimizers/ .
                        Whereas recently the paper "Attention is All You Need"
                        suggested a value of 0.98 for beta2, this parameter may
                        not work well for normal models / default
                        baselines.""")
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help="""Label smoothing value epsilon.
                        Probabilities of all non-true labels
                        will be smoothed by epsilon / (vocab_size - 1).
                        Set to zero to turn off label smoothing.
                        For more detailed information, see:
                        https://arxiv.org/abs/1512.00567""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) steps have gone past
                        start_decay_steps""")
    parser.add_argument('-start_decay_steps', type=int, default=50000,
                        help="""Start decaying every decay_steps after
                        start_decay_steps""")
    parser.add_argument('-decay_steps', type=int, default=10000,
                        help="""Decay every decay_steps""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=[''], help="Use a custom decay rate. noam not supported")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    # Saving and Logging
    parser.add_argument('-save_dir', default='./save_dir', type=str)
    parser.add_argument('-logging_steps', default=50, type=int,
                        help='logging frequency')
    parser.add_argument('-checkpoint_steps', default=100, type=int,
                        help='checkpoint frequency')

    opt = parser.parse_args()
    trainer = Trainer(opt)
