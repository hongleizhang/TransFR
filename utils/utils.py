import os
import datetime
import pickle

import numpy as np

import torch


# from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logs(log_path: str = 'logs'):
    # logger = SummaryWriter(log_path)
    logger = None
    return logger


def dump(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))
    pass


def load(filename):
    f = open(filename, 'rb')
    model = pickle.load(f)
    return model


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len - 3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize(rating, minVal=0, maxVal=5):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return float(rating - minVal) / (maxVal - minVal) + 0.01
    elif maxVal == minVal:
        return rating / maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError


def denormalize(rating, minVal=0, maxVal=5):
    return minVal + (rating - 0.01) * (maxVal - minVal)
