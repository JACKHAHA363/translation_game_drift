""" Pretrain the IWSLT
"""
import torch
import argparse
import os
from ld_research.pretrain.settings import EXP_FOLDER
from onmt.train_single import opts, training_opt_postprocessing, init_logger, logger, lazily_load_dataset, \
    _load_fields, _collect_report_features, build_model, _tally_parameters, _check_save_model_path, build_optim, \
    build_trainer, build_model_saver, build_dataset_iter

def modify_opt(opt):
    """ modify opt to match the paper """
    # Emb Options
    opt.src_word_vec_size = 256
    opt.tgt_word_vec_size = 256
    opt.word_vec_size = 256
    opt.share_decoder_embeddings = True
    opt.share_embeddings = False

    # Encoder-Decoder Options
    opt.layers = 1
    opt.enc_layers = 1
    opt.dec_layers = 1
    opt.rnn_size = 256
    opt.enc_rnn_size = 256
    opt.dec_rnn_size = 256
    opt.rnn_type = 'GRU'

    # Save and log
    opt.tensorboard = True
    opt.tensorboard_log_dir = os.path.join(EXP_FOLDER, 'logs')
    opt.save_model = os.path.join(EXP_FOLDER, 'model')
    logger.info('Save results to %s' % EXP_FOLDER)
    return opt

def main(opt, device_id=None):
    """ Main logic. Copy of train_single """
    opt = training_opt_postprocessing(opt, device_id)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.
    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt, is_train=False)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='pretrain IWSLT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()
    init_logger(opt.log_file)
    opt = modify_opt(opt)

    # CPU
    main(opt, -1)

    ## GPU
    #main(opt, 0)
