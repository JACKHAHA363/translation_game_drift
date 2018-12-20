""" Utilities to prepare parser
"""
import argparse

from ld_research.settings import FR, EN, DE


def add_pretrain_args(parser):
    """ Args related to pretrain """
    parser.add_argument('-src_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    parser.add_argument('-tgt_lang', type=str, required=True,
                        choices=[FR, EN, DE],
                        help='The source language')
    return parser


def add_communication_args(parser):
    """ Add argument specific for communication """
    parser.add_argument('-pretrain_fr_en_ckpt', default=None, type=str,
                        help='path to the fr-en checkpoint')
    parser.add_argument('-pretrain_en_de_ckpt', default=None, type=str,
                        help='path to the en-de checkpoint')
    parser.add_argument('-v_coeff', default=0.5, type=float,
                        help='value loss coefficient')
    parser.add_argument('-ent_coeff', default=0.01, type=float,
                        help='entropy loss coefficient')
    parser.add_argument('-reduce_ent', action='store_true',
                        help='If true, the entropy loss is inverted to reduce the entropy')
    parser.add_argument('-norm_reward', action='store_true',
                        help='Whether or not to normalize the reward for sentence length')
    parser.add_argument('-disable_dropout', action='store_true',
                        help='Disable the dropout of both agents and value net when training.')
    return parser


def add_agent_args(parser):
    """ Args for value network and agents """
    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-value_emb_size', default=256, type=int)
    parser.add_argument('-value_hidden_size', default=256, type=int)
    parser.add_argument('-param_init', default=0.1, type=float,
                        help='model initialization range')
    parser.add_argument('-device', default='cpu', type=str,
                        help='The device. None, cpu, cuda, cuda/cpu:rank')
    parser.add_argument('-dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    return parser


def add_general_args(parser):
    """ General arguments """
    parser.add_argument('-save_dir', required=True, type=str)
    parser.add_argument('-logging_steps', default=2, type=int,
                        help='logging frequency')
    parser.add_argument('-checkpoint_steps', default=100, type=int,
                        help='checkpoint frequency')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size for training')
    parser.add_argument('-valid_steps', type=int, default=10000,
                        help='Perfom validation every X steps')
    parser.add_argument('-train_steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help="""Label smoothing value epsilon.
                        Probabilities of all non-true labels
                        will be smoothed by epsilon / (vocab_size - 1).
                        Set to zero to turn off label smoothing.
                        For more detailed information, see:
                        https://arxiv.org/abs/1512.00567""")
    parser.add_argument('-sample_method', choices=['random', 'greedy'],
                        type=str, default='greedy', help='Which method to translate')
    return parser


def add_optimizer_args(parser, prefix=''):
    """ Add optimziation args with prefix """
    # Method
    parser.add_argument('-{}optim'.format(prefix), default='sgd',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-{}adagrad_accumulator_init'.format(prefix), type=float, default=0,
                        help="""Initializes the accumulator values in adagrad.
                        Mirrors the initial_accumulator_value option
                        in the tensorflow adagrad (use 0.1 for their default).
                        """)
    parser.add_argument('-{}max_grad_norm'.format(prefix), type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-{}adam_beta1'.format(prefix), type=float, default=0.9,
                        help="""The beta1 parameter used by Adam.
                        Almost without exception a value of 0.9 is used in
                        the literature, seemingly giving good results,
                        so we would discourage changing this value from
                        the default without due consideration.""")
    parser.add_argument('-{}adam_beta2'.format(prefix), type=float, default=0.999,
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
    return parser


def add_lr_schedule_args(parser, prefix=''):
    """ Add arguments for learning schedule """
    parser.add_argument('-{}learning_rate'.format(prefix), type=float, default=1.0,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-{}learning_rate_decay'.format(prefix), type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) steps have gone past
                        start_decay_steps""")
    parser.add_argument('-{}start_decay_steps'.format(prefix), type=int, default=50000,
                        help="""Start decaying every decay_steps after
                        start_decay_steps""")
    parser.add_argument('-{}decay_steps'.format(prefix), type=int, default=10000,
                        help="""Decay every decay_steps""")
    parser.add_argument('-{}decay_method'.format(prefix), type=str, default="piecewise",
                        choices=['piecewise'], help="Use a custom decay rate. noam not supported")
    parser.add_argument('-{}warmup_steps'.format(prefix), type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")
    return parser


def get_pretrain_parser():
    """ Return the args parser used for pretrain task """
    parser = argparse.ArgumentParser('Pretrain')
    parser = add_general_args(parser)
    parser = add_pretrain_args(parser)
    parser = add_agent_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_lr_schedule_args(parser)
    return parser


def get_communicate_parser():
    """ Return the parser for communication task """
    parser = argparse.ArgumentParser('Finetune PG')
    parser = add_general_args(parser)
    parser = add_communication_args(parser)
    parser = add_agent_args(parser)

    parser = add_optimizer_args(parser, 'value_')
    parser = add_lr_schedule_args(parser, 'value_')

    parser = add_optimizer_args(parser, 'fr_en_')
    parser = add_lr_schedule_args(parser, 'fr_en_')

    parser = add_optimizer_args(parser, 'en_de_')
    parser = add_lr_schedule_args(parser, 'en_de_')
    return parser


def get_communicate_lm_parser():
    """ Return the parser for finetune with LM """
    parser = get_communicate_parser()
    parser.add_argument('-lm_ckpt', required=True, type=str,
                        help='path to pretrained language model')
    parser.add_argument('-lm_coeff', default=0.5, type=float,
                        help='coefficient of englishness reward')
    return parser


def get_language_model_parser():
    """ share the pretrain_parser """
    parser = argparse.ArgumentParser('LM pretrain')
    parser = add_general_args(parser)
    parser = add_agent_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_lr_schedule_args(parser)
    return parser


def get_communicate_ppo_parser():
    """ Return the parser for finetune with PPO """
    parser = get_communicate_parser()
    parser.add_argument('-mini_bsz', default=64, type=int,
                        help='mini batch size of PPO')
    parser.add_argument('-ppo_epochs', default=2, type=int,
                        help='The number of PPO inner epochs')
    parser.add_argument('-clip_param', default=0.1, type=float,
                        help='The clipping of importance ratio')
    return parser
