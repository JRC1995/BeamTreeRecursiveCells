import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="LanmguageProcessor Hyperparameter-Tuning Arguments")
    parser.add_argument('--model', type=str, default="Transformer",
                        choices=["BalancedTreeCell", "GumbelTreeLSTM", "RCell"])
    parser.add_argument('--no_display', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="sentence_pair",
                        choices=["sentence_pair", "classifier", "seq_label"])
    parser.add_argument('--dataset', type=str, default="proplogic",
                        choices=["proplogic", "proplogic_C", "SST2", "SST5", "conll2003", "ontonotes5_ner", "MNLIdev"])
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_hypercheckpoint', type=str2bool, default=False)
    parser.add_argument('--load_checkpoint', type=str2bool, default=False)
    parser.add_argument('--allow_repeat', type=str2bool, default=False)
    parser.add_argument('--reproducible', type=str2bool, default=True)
    return parser
