"""
Entry point for running an experiment with SDNet.
"""
import argparse
import os
import logging
from config import Configuration
from loaders import loader_factory
from sdnet import SDNet
from sdnet_trainer import SDNetTrainer


def init_logging(config):
    if not os.path.exists(config.folder):
        os.makedirs(config.folder)
    logging.basicConfig(filename=config.folder + '/logfile.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    log = logging.getLogger()
    log.debug(config.__dict__)
    log.info('---- Setting up experiment at ' + config.folder + '----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SDNet')
    parser.add_argument('--epochs', help='Number of epochs to train', type=int)
    parser.add_argument('--dataset', help='Dataset to use', choices=['acdc'], required=True)
    parser.add_argument('--description', help='Experiment description')
    parser.add_argument('--split', help='Split for Cross Validation', type=int, required=True)
    parser.add_argument('--test', help='Test', type=bool)
    parser.add_argument('--ul_mix', help='Percentage of unlabelled data to mix', type=float, required=True)
    parser.add_argument('--l_mix', help='Percentage of labelled data to mix', type=float, required=True)
    args = parser.parse_args()

    # Create configuration object from parameters
    loader = loader_factory.init_loader(args.dataset)
    data = loader.load_labelled_data(args.split, 'training')

    folder = 'sdnet_%s_ul_%.3f_l_%.3f_split%d' % (args.dataset, args.ul_mix, args.l_mix, args.split)
    conf = Configuration(folder, data.size(), data.shape()[1:])
    del data

    conf.description = args.description if args.description else ''
    if args.epochs:
        conf.epochs = args.epochs
    conf.dataset_name = args.dataset
    conf.ul_mix = args.ul_mix
    conf.l_mix = args.l_mix
    conf.split = args.split
    conf.save()

    init_logging(conf)

    sdnet = SDNet(conf)
    sdnet.build()

    trainer = SDNetTrainer(sdnet, conf)

    if not args.test:
        trainer.fit()
    trainer.test()
