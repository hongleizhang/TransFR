import logging
import os

import psutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# warnings.filterwarnings('ignore')

from ClientManager import ClientManager
from ServerManager import ServerManager
from TrainManager import TrainManager
from options import args_parser
from utils import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    utils.set_seed(seed=42)

    args = args_parser()
    args_prient = utils.format_arg_str(args, exclude_lst=['verbose'])
    logging.info(args_prient)
    configs = vars(args)

    # Server
    serverManager = ServerManager(configs)

    # Client
    clientManager = ClientManager(configs)

    # Training
    trainManager = TrainManager(configs, serverManager, clientManager)
    trainManager.train_federated()

    print(u'current memory overhead：%.4f GB' % (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    logging.info('-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    main()
