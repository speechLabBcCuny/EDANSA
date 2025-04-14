""" command line args for running experiments
"""

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        help=
        'runs a debug run with a small number of epochs and a small number of samples',
        action='store_true')
    parser.add_argument('--project_name', type=str, help='project name')
    parser.add_argument('--dataset_name_v', type=str, help='dataset name')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--patience',
                        type=int,
                        help='patience for early stopping')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--device', type=int, help='device')
    parser.add_argument('--run_id_2resume', type=str, help='run id to resume')
    parser.add_argument('--checkpointfile_2resume',
                        type=str,
                        help='checkpoint file to resume')
    parser.add_argument('--checkpoint_every_Nth_epoch',
                        type=int,
                        help='checkpoint every Nth epoch')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--run-name', type=str, help='Name of the WandB run')
    parser.add_argument('--dataset_csv_path', type=str, help='dataset csv path')
    parser.add_argument('--dataset_in_memory',
                        type=str2bool,
                        help='dataset in memory',
                        default=None)
    parser.add_argument('--wandb_mode',
                        type=str,
                        help='wandb mode',
                        default=None,
                        choices=['online', 'offline', 'disabled'])

    return parser
