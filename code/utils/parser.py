"""Conditional ArgumentParser class enabling easy loading and saving of human-readable
configuration files, dictionary arguments editing through command-line, and conditional
list of arguments.

See https://github.com/aRI0U/python-parser for more details
"""

import argparse
import json
from pathlib import Path
import re


class Parser:
    def __init__(self, **kwargs):

        cli_parser = argparse.ArgumentParser(add_help=False)

        cli = cli_parser.add_argument_group('command line arguments')

        # FIRST ORDER OPTIONS

        cli.add_argument('-c', '--config', type=Path, metavar='PATH',
                         help='configuration file')
        cli.add_argument('-d', '--debug', action='store_true',
                         help='debug mode')
        cli.add_argument('-l', '--load_model', type=Path, metavar='PATH',
                         help='model to load')
        cli.add_argument('-t', '--test', action='store_true',
                         help='test reconstructions')

        parser = argparse.ArgumentParser(
            parents=[cli_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=True,
            **kwargs
        )

        # GROUP OF ARGUMENTS
        dataset = parser.add_argument_group('dataset-related options')
        network = parser.add_argument_group('architecture of the model')
        hparams = parser.add_argument_group('hyperparameters')
        diverse = parser.add_argument_group('miscellaneous')

        # MAIN ARGUMENTS

        args, _ = cli_parser.parse_known_args()
        dataset.add_argument('--dataset_name', type=str, choices=['MNIST', 'NSynth'], default='MNIST',
                             help='name of the dataset used')
        dataset.add_argument('--dl_kwargs', type=dict,
                             help='dataloader keyword arguments (batch size, num_workers, etc.)')
        dataset.add_argument('--input_dim', type=int, default=1,
                             help='number of channels of the encoder input')

        network.add_argument('--architecture', type=str, choices=['convnet', 'convnet2', 'mnist', 'mnist_like'], default='mnist',
                             help='architecture used for encoder and decoder')
        network.add_argument('--ema', action='store_true',
                             help='use Exponential Moving Averages for updating latent embeddings')
        network.add_argument('--ema_decay', type=float, default=0.95,
                             help='smoothness coefficient for EMA updates')
        network.add_argument('--normalize', action='store_true',
                             help='normalize spectrograms before feeding them to the network')
        network.add_argument('--restarts', action='store_true',
                             help='resample latent vectors when they are not used enough')

        hparams.add_argument('--adam', type=dict, default={},
                             help='hyperparameters of the main optimizer')
        hparams.add_argument('--commitment_cost', type=float, default=0.15,
                             help='weight for the commitment cost')
        hparams.add_argument('--latent_dim', type=int, default=16,
                             help='dimension of the latent space')
        hparams.add_argument('--num_codewords', type=int, default=16,
                             help='number of codewords used as discrete embeddings')
        hparams.add_argument('--nfft', type=int, default=512,
                             help='Nfft for STFT')
        hparams.add_argument('--win_length', type=int, default=512,
                             help='number of timesteps per audio sample')

        diverse.add_argument('--gpus', type=int, nargs='*',
                             help='GPUs the model is loaded on')

        args.train = not args.test

        # CONDITIONAL ARGUMENTS
        if args.train:
            train = parser.add_argument_group('training options')

            train.add_argument('--max_epochs', type=int, default=50,
                               help='number of epochs of training')
            train.add_argument('--save_freq', type=int, default=5,
                               help='frequency checkpoints and results are saved')

        else:
            test = parser.add_argument_group('testing options')

        if args.load_model and (args.load_model / 'config.json').exists():
            args.config = args.load_model / 'config.json'

        if args.config:
            # loading configuration file
            with open(args.config, 'r') as cfg_file:
                parser.set_defaults(**json.load(cfg_file))

        self.parser = parser

        # arguments from those groups should be saved in the config file of the experiments
        self.groups_to_save = [dataset, network, hparams]

    def __repr__(self):
        return repr(self.parser)

    def parse_args(self, *args, **kwargs):
        r"""Parse the arguments of the program
        """
        args, unknown = self.parser.parse_known_args(*args, **kwargs)
        args.train = not args.test

        dict, key = None, None
        for arg in unknown:
            match = re.match('-+(.+?)_(.+?)\Z', arg)  # detect options of kind --sth_sth
            if match is not None:
                dict, key = match.group(1), match.group(2)
            else:
                try:
                    getattr(args, dict)[key] = self.infer_type(arg)
                except AttributeError:
                    self.parser.print_usage()
                    print(f"{self.parser.prog}: error: unrecognized arguments: --{dict}_{key}")
                    exit(2)
                except TypeError:
                    self.parser.print_usage()
                    print(f"{self.parser.prog}: error: unrecognized arguments:", arg)
                    exit(2)
        return args

    def save(self, args, path):
        r"""Save the parsed options
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        args_dict = {}

        for group in self.groups_to_save:
            args_dict.update({arg.dest: getattr(args, arg.dest) for arg in group._group_actions})

        with open(path / 'config.json', 'w') as fp:
            json.dump(
                args_dict,
                fp,
                indent=4,
                default=lambda arg: str(arg),
                sort_keys=True
            )

    @staticmethod
    def infer_type(s):
        r"""Converts a string into the most probable type:
            - int if it is composed of digits only
            - float if it can be converted to float
            - str otherwise
        """
        if s.isdigit():
            return int(s)
        try:
            return float(s)
        except ValueError:
            return s
