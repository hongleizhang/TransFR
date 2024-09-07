import math
from copy import deepcopy

import numpy as np
import torch

from reader.data_reader import DataReader


class ServerManager:
    def __init__(self, configs: dict):
        self.configs = configs
        print('Preparing for processing and reading data...')
        self.dr = DataReader(self.configs)
        self.configs['num_users'], self.configs['num_items'] = self.get_user_item_nums
        print(f'The number of users and items is : {self.get_user_item_nums}')
        self.server = self.get_server_instance()
        print(f'Global model at Server:\t {self.server.global_model}')

    def get_server_instance(self):
        server = Server(self.configs)
        server_global_model = self.init_server_global_model()
        server.set_server_global_model(server_global_model)
        return server

    def init_server_global_model(self):
        model_name = self.configs['model'] + '.' + self.configs['model']
        server_global_model = eval(model_name)(self.configs, init_type='server').to(self.configs['device'])
        return server_global_model

    def aggregate_adapter_from_clients(self, adapters, clients_len):
        with torch.no_grad():
            for params in zip(*[model.parameters() for model in adapters]):
                summed_params = torch.sum(torch.stack(params), dim=0)
        summed_params = summed_params / clients_len
        for param, summed_param in zip(self.server.global_model.items_emb.adapter.parameters(), summed_params):
            param.data.copy_(summed_param)
        pass

    def get_latest_adapter(self):
        adapter = deepcopy(self.server.global_model.items_emb.adapter)
        return adapter

    @property
    def get_client_list(self):
        return self.dr.get_client_list

    @property
    def get_random_client_list(self):
        size = math.ceil(len(self.get_client_list) * self.configs['client_frac'])
        random_client_list = np.random.choice(self.get_client_list, size)
        return random_client_list

    @property
    def get_test_client_list(self):
        random_client_list = np.array([0, 1, 2, 3, 4])
        return random_client_list

    @property
    def get_item_list(self):
        return self.dr.get_item_list

    @property
    def get_user_item_nums(self):
        return self.dr.get_user_item_nums


class Server:

    def __init__(self, configs: dict):
        self.configs = configs
        self.global_model = None

    def set_server_global_model(self, global_model):
        self.global_model = global_model
        pass


if __name__ == "__main__":
    configs = {
        'dataset': 'filmtrust',
        'data_type': 'implicit',
        'num_negative_train': 4,
        'num_negative_test': 99,
        'local_batch_size': 100,
        'top_k': 10
    }

    server = Server(configs=configs)
    print(server.get_client_list)
