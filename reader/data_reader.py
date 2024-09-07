# encoding:utf-8
import sys

import psutil

sys.path.append("..")
import os
import numpy as np
import pandas as pd
from utils import utils


class DataReader(object):
    """
    Load data from disk and return clients' data
    """

    def __init__(self, configs: dict):
        super(DataReader, self).__init__()
        self.configs = configs
        assert '-' in self.configs['dataset'], "please use Book-Movie, Music-Movie, Music-Book"
        self.source_dataset, self.target_dataset = self.configs['dataset'].split('-')
        self.current_data_path = 'data/processed/' + self.configs['dataset']

        if not os.path.exists(self.current_data_path):
            os.makedirs(self.current_data_path)
            os.makedirs(self.current_data_path + '/' + self.source_dataset)
            os.makedirs(self.current_data_path + '/' + self.target_dataset)

        if os.path.exists(self.current_data_path + '/' + self.source_dataset + '/raw_data_all_' + self.configs[
            'data_type'] + '.pkl'):
            self.__raw_data_all = utils.load(
                self.current_data_path + '/' + self.source_dataset + '/raw_data_all_' + self.configs[
                    'data_type'] + '.pkl')
            self.__raw_text_data_all = utils.load(
                self.current_data_path + '/' + self.source_dataset + '/raw_text_data_all_' + self.configs[
                    'data_type'] + '.pkl')
            self.__raw_data_all_target = utils.load(
                self.current_data_path + '/' + self.target_dataset + '/raw_data_all_target_' + self.configs[
                    'data_type'] + '.pkl')
            self.__raw_text_data_all_target = utils.load(
                self.current_data_path + '/' + self.target_dataset + '/raw_text_data_all_target_' + self.configs[
                    'data_type'] + '.pkl')
        else:
            self.__raw_data_all, self.__raw_text_data_all = self.__read_raw_data(self.source_dataset)
            self.__raw_data_all_target, self.__raw_text_data_all_target = self.__read_raw_data(self.target_dataset)
            utils.dump(self.__raw_data_all,
                       self.current_data_path + '/' + self.source_dataset + '/raw_data_all_' + self.configs[
                           'data_type'] + '.pkl')
            utils.dump(self.__raw_text_data_all,
                       self.current_data_path + '/' + self.source_dataset + '/raw_text_data_all_' + self.configs[
                           'data_type'] + '.pkl')

            utils.dump(self.__raw_data_all_target,
                       self.current_data_path + '/' + self.target_dataset + '/raw_data_all_target_' + self.configs[
                           'data_type'] + '.pkl')
            utils.dump(self.__raw_text_data_all_target,
                       self.current_data_path + '/' + self.target_dataset + '/raw_text_data_all_target_' + self.configs[
                           'data_type'] + '.pkl')

        self.rating_min = self.__raw_data_all['ratings'].min()
        self.rating_max = self.__raw_data_all['ratings'].max()

        self.centralized_data, self.centralized_text_data = self.get_data_centralized(self.source_dataset)
        self.centralized_data_train_group = self.centralized_data['train'].groupby(by='user_id')
        self.centralized_data_val_group = self.centralized_data['val'].groupby(by='user_id')
        self.centralized_data_test_group = self.centralized_data['test'].groupby(by='user_id')

        self.centralized_data_target, self.centralized_text_data_target = self.get_data_centralized(self.target_dataset)
        self.centralized_data_train_group_target = self.centralized_data_target['train'].groupby(by='user_id')
        self.centralized_data_val_group_target = self.centralized_data_target['val'].groupby(by='user_id')
        self.centralized_data_test_group_target = self.centralized_data_target['test'].groupby(by='user_id')

    @property
    def get_data_length(self):
        return len(self.__raw_data_all)

    @property
    def get_data_rating_range(self):
        return self.rating_min, self.rating_max

    @property
    def get_user_item_nums(self):
        user_num, item_num = self.__raw_data_all['user_id'].max() + 1, self.__raw_data_all['item_id'].max() + 1
        return user_num, item_num

    @property
    def get_client_list(self):
        client_list = self.__raw_data_all['user_id'].unique()
        return client_list

    @property
    def get_item_list(self):
        item_list = self.__raw_data_all['item_id'].unique()
        return item_list

    # for each client
    def get_data_by_client(self, client_id: int) -> pd.DataFrame:
        assert client_id in self.get_client_list, "not exist in datasets"
        client_data = {}
        client_data['train'] = self.centralized_data_train_group.get_group(name=client_id)
        client_data['val'] = self.centralized_data_val_group.get_group(name=client_id)
        client_data['test'] = self.centralized_data_test_group.get_group(name=client_id)
        client_data['target_train'] = self.centralized_data_train_group_target.get_group(name=client_id)
        client_data['target_val'] = self.centralized_data_val_group_target.get_group(name=client_id)
        client_data['target_test'] = self.centralized_data_test_group_target.get_group(name=client_id)
        return client_data

    # for all clients for centralized training
    def get_data_centralized(self, dataset):

        all_data = {}
        read_flag = True

        central_trian_path = self.current_data_path + '/' + dataset + '/' + dataset + '_train.csv'
        central_val_path = self.current_data_path + '/' + dataset + '/' + dataset + '_val.csv'
        central_test_path = self.current_data_path + '/' + dataset + '/' + dataset + '_test.csv'

        data_train, data_val, data_test = pd.DataFrame(columns=['user_id', 'item_id', 'ratings']), pd.DataFrame(
            columns=['user_id', 'item_id', 'ratings']), pd.DataFrame(columns=['user_id', 'item_id', 'ratings'])

        if os.path.exists(central_trian_path):
            data_train = pd.read_csv(central_trian_path)
            read_flag = False
        if os.path.exists(central_val_path):
            data_val = pd.read_csv(central_val_path)
            read_flag = False
        if os.path.exists(central_test_path):
            data_test = pd.read_csv(central_test_path)
            read_flag = False

        if read_flag and dataset == self.source_dataset:
            self.__data_by_clients = self.__raw_data_all.groupby(by='user_id')
            clients_data = [client_data for client_id, client_data in self.__data_by_clients]

            result = map(self.__get_train_val_test_data_client, clients_data)
            for i, x in enumerate(result):
                client_train, client_val, client_test = x

                data_train = pd.concat([data_train, client_train])
                data_val = pd.concat([data_val, client_val])
                data_test = pd.concat([data_test, client_test])

            data_train.to_csv(central_trian_path, header=['user_id', 'item_id', 'ratings'], index=False)
            data_val.to_csv(central_val_path, header=['user_id', 'item_id', 'ratings'], index=False)
            data_test.to_csv(central_test_path, header=['user_id', 'item_id', 'ratings'], index=False)

        if read_flag and dataset == self.target_dataset:
            self.__data_by_clients = self.__raw_data_all_target.groupby(by='user_id')
            clients_data = [client_data for client_id, client_data in self.__data_by_clients]

            result = map(self.__get_train_val_test_data_client, clients_data)
            for i, x in enumerate(result):
                client_train, client_val, client_test = x

                data_train = pd.concat([data_train, client_train])
                data_val = pd.concat([data_val, client_val])
                data_test = pd.concat([data_test, client_test])

            data_train.to_csv(central_trian_path, header=['user_id', 'item_id', 'ratings'], index=False)
            data_val.to_csv(central_val_path, header=['user_id', 'item_id', 'ratings'], index=False)
            data_test.to_csv(central_test_path, header=['user_id', 'item_id', 'ratings'], index=False)

        all_data['train'] = data_train
        all_data['val'] = data_val
        all_data['test'] = data_test

        all_text_data = None
        if dataset == self.source_dataset:
            all_text_data = self.__raw_text_data_all
        else:
            all_text_data = self.__raw_text_data_all_target

        return all_data, all_text_data

    def __get_train_val_test_data_client(self, client_data: pd.DataFrame):

        user_id = int(client_data.head(1)['user_id'])
        item_list = np.arange(self.get_user_item_nums[1])

        client_train, client_val, client_test = None, None, None

        if self.configs['data_type'] == 'implicit':

            if len(client_data) < 3:
                return client_data, pd.DataFrame(
                    columns=['user_id', 'item_id', 'ratings']), pd.DataFrame(
                    columns=['user_id', 'item_id', 'ratings'])  # val and test is empty dataframe

            client_test = client_data.tail(1).copy()
            client_data = client_data.drop(client_test.index)
            client_val = client_data.tail(1).copy()
            client_data = client_data.drop(client_val.index)

            negative_len_train = self.configs['num_negative_train'] * len(client_data)
            user_id_seq = np.array([user_id] * negative_len_train)
            train_neg_items = np.random.choice(item_list, negative_len_train)
            train_neg_label = np.array([0] * negative_len_train)

            train_negatives = pd.DataFrame(zip(user_id_seq, train_neg_items, train_neg_label),
                                           columns=['user_id', 'item_id', 'ratings'])

            client_train = pd.concat([client_data, train_negatives], axis=0).reset_index(drop=True)

            negative_len_val = self.configs['num_negative_test']
            val_neg_items = np.random.choice(item_list, negative_len_val)
            user_id_seq = np.array([user_id] * negative_len_val)
            val_neg_label = np.array([0] * negative_len_val)

            val_negatives = pd.DataFrame(zip(user_id_seq, val_neg_items, val_neg_label),
                                         columns=['user_id', 'item_id', 'ratings'])

            client_val = pd.concat([client_val, val_negatives], axis=0).reset_index(drop=True)

            negative_len_test = self.configs['num_negative_test']
            test_neg_items = np.random.choice(item_list, negative_len_test)
            user_id_seq = np.array([user_id] * negative_len_test)
            test_neg_label = np.array([0] * negative_len_test)

            test_negatives = pd.DataFrame(zip(user_id_seq, test_neg_items, test_neg_label),
                                          columns=['user_id', 'item_id', 'ratings'])

            client_test = pd.concat([client_test, test_negatives], axis=0).reset_index(drop=True)

            return client_train, client_val, client_test

        else:

            train_ratio, val_ratio, test_ratio = [0.8, 0.1, 0.1]
            len_by_user = len(client_data)

            val_num = int(len_by_user * val_ratio)
            test_num = int(len_by_user * test_ratio)
            train_num = len_by_user - test_num - val_num

            client_train = client_data[:train_num].reset_index(drop=True)
            client_val = client_data[train_num:train_num + val_num].reset_index(drop=True)
            client_test = client_data[train_num + val_num:].reset_index(drop=True)
            # client_train['ratings'] = client_train['ratings'].apply(utils.normalize,
            #                                                         args=(self.rating_min, self.rating_max,))
            # client_val['ratings'] = client_val['ratings'].apply(utils.normalize,
            #                                                     args=(self.rating_min, self.rating_max,))
            # client_test['ratings'] = client_test['ratings'].apply(utils.normalize,
            #                                                       args=(self.rating_min, self.rating_max,))

            return client_train, client_val, client_test

    def __read_raw_data(self, dataset):
        '''
        read raw data
        :param data_name:
        :return:
        '''
        data_name = self.configs['dataset']
        assert data_name in ['Book-Movie', 'Music-Movie',
                             'Music-Book'], "please use Book-Movie, Music-Movie, Music-Book"
        raw_data_all, raw_text_data_all = None, None

        if '-' in data_name:
            data_path = 'data/raw/' + self.configs['dataset'] + '/' + dataset + '/rating_data.csv'
            raw_data_all = pd.read_csv(data_path, sep=',', header=0, engine='python',
                                       names=['user_id', 'item_id', 'ratings', 'time_stamp'])
            data_path = 'data/raw/' + self.configs['dataset'] + '/' + dataset + '/meta_data.csv'
            raw_text_data_all = pd.read_csv(data_path, sep=',', header=0, engine='python',
                                            names=['item_id', 'description', 'title', 'categories'])

        raw_data_all = raw_data_all.drop_duplicates().reset_index(drop=True)

        if self.configs['cold_nums'] > 0:
            raw_data_all = raw_data_all.groupby('user_id').filter(lambda x: len(x) >= self.configs['cold_nums'])

        data_all, text_data_all = self.__remapping_users_items_text(raw_data_all, raw_text_data_all)

        data_all = data_all.groupby('user_id', group_keys=False).apply(lambda x: x.sort_values('time_stamp'))

        data_all = data_all.drop(columns=['time_stamp'])

        if self.configs['data_type'] == 'implicit':
            if data_all['ratings'].min() > 0:
                data_all['ratings'] = 1.0
            else:
                data_all['ratings'][data_all['ratings'] > 0] = 1.0

        # text_data_all['item_texts_total'] = text_data_all['categories'] + ' ' + text_data_all['title'] + ' ' + text_data_all['description']
        text_data_all['item_texts_total'] = text_data_all.apply(
            lambda row: '; '.join([str(row['categories']), str(row['title']), str(row['description'])]), axis=1)
        text_data_all = text_data_all.drop(columns=['categories'])
        text_data_all = text_data_all.drop(columns=['title'])
        text_data_all = text_data_all.drop(columns=['description'])

        text_data_all['item_texts'] = text_data_all.apply(lambda row: row['item_texts_total'][:512], axis=1)

        text_to_path = 'data/processed/' + self.configs['dataset'] + '/' + dataset + '/' + dataset + '_text.csv'
        text_data_all.to_csv(text_to_path, header=['item_id', 'item_texts', 'item_texts_total'], index=False)

        return data_all, text_data_all

    def __remapping_users_items(self, raw_data_all: pd.DataFrame):

        users_raw_id, items_raw_id = sorted(raw_data_all['user_id'].unique()), sorted(raw_data_all['item_id'].unique())
        users_len, items_len = len(users_raw_id), len(items_raw_id)

        users_dict = dict(zip(users_raw_id, range(0, users_len)))
        items_dict = dict(zip(items_raw_id, range(0, items_len)))

        raw_data_all['user_id'] = raw_data_all['user_id'].map(users_dict)
        raw_data_all['item_id'] = raw_data_all['item_id'].map(items_dict)

        return raw_data_all

    def __remapping_users_items_text(self, raw_data_all: pd.DataFrame, raw_text_data_all: pd.DataFrame):

        users_raw_id, items_raw_id = sorted(raw_data_all['user_id'].unique()), sorted(raw_data_all['item_id'].unique())
        users_len, items_len = len(users_raw_id), len(items_raw_id)

        users_dict = dict(zip(users_raw_id, range(0, users_len)))
        items_dict = dict(zip(items_raw_id, range(0, items_len)))

        raw_data_all['user_id'] = raw_data_all['user_id'].map(users_dict)
        raw_data_all['item_id'] = raw_data_all['item_id'].map(items_dict)
        raw_text_data_all['item_id'] = raw_text_data_all['item_id'].map(items_dict)
        raw_text_data_all.dropna(subset=['item_id'], inplace=True)
        # raw_text_data_all.dropna(subset=['item_id'], inplace=False)
        raw_text_data_all['item_id'] = raw_text_data_all['item_id'].astype(int)

        return raw_data_all, raw_text_data_all


if __name__ == "__main__":
    configs = {
        'dataset': 'ml-1m',
        'data_type': 'explicit',
        'cold_nums': 10,
        'num_negative_train': 4,
        'num_negative_test': 99
    }
    dr = DataReader(configs)
    dr.get_train_val_test_data()
    # data = dr.get_data_by_client(0)

    print(u'current memory overhead：%.4f GB' % (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
