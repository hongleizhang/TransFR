import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TDataLoader

from reader.data_reader import DataReader


class DataLoader():
    def __init__(self, configs, client_data, text_data=None, text_data_target=None):
        self.configs = configs
        self.train_data, self.val_data, self.test_data, self.train_data_target, self.val_data_target, self.test_data_target = \
        client_data['train'], client_data['val'], client_data[
            'test'], client_data['target_train'], client_data['target_val'], client_data['target_test']
        self.text_data = text_data
        self.text_data_target = text_data_target

    def get_train_dataloader(self):

        users, items, item_texts, labels = torch.LongTensor(np.array(self.train_data['user_id'])), torch.LongTensor(
            np.array(self.train_data['item_id'])), self.text_data, torch.FloatTensor(
            np.array(self.train_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        return TDataLoader(dataset, batch_size=self.configs['local_batch_size'], shuffle=True)

    def get_val_dataloader(self):

        if self.val_data.empty:
            users, items, item_texts, labels = torch.LongTensor(self.val_data['user_id']), torch.LongTensor(
                self.val_data['item_id']), self.text_data, torch.FloatTensor(self.val_data['ratings'])
        else:
            users, items, item_texts, labels = torch.LongTensor(np.array(self.val_data['user_id'])), torch.LongTensor(
                np.array(self.val_data['item_id'])), self.text_data, torch.FloatTensor(
                np.array(self.val_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        client_data_len = len(items)

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)

    def get_test_dataloader(self):

        if self.test_data.empty:
            users, items, item_texts, labels = torch.LongTensor(self.test_data['user_id']), torch.LongTensor(
                self.test_data['item_id']), self.text_data, torch.FloatTensor(self.test_data['ratings'])
        else:
            users, items, item_texts, labels = torch.LongTensor(np.array(self.test_data['user_id'])), torch.LongTensor(
                np.array(self.test_data['item_id'])), self.text_data, torch.FloatTensor(
                np.array(self.test_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        client_data_len = len(items)

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)

    def get_train_target_dataloader(self):

        users, items, item_texts, labels = torch.LongTensor(
            np.array(self.train_data_target['user_id'])), torch.LongTensor(
            np.array(self.train_data_target['item_id'])), self.text_data_target, torch.FloatTensor(
            np.array(self.train_data_target['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        return TDataLoader(dataset, batch_size=self.configs['local_batch_size'], shuffle=True)

    def get_val_target_dataloader(self):

        if self.val_data.empty:
            users, items, item_texts, labels = torch.LongTensor(self.val_data_target['user_id']), torch.LongTensor(
                self.val_data_target['item_id']), self.text_data_target, torch.FloatTensor(
                self.val_data_target['ratings'])
        else:
            users, items, item_texts, labels = torch.LongTensor(
                np.array(self.val_data_target['user_id'])), torch.LongTensor(
                np.array(self.val_data_target['item_id'])), self.text_data_target, torch.FloatTensor(
                np.array(self.val_data_target['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        client_data_len = len(items)

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)

    def get_test_target_dataloader(self):

        if self.test_data.empty:
            users, items, item_texts, labels = torch.LongTensor(self.test_data_target['user_id']), torch.LongTensor(
                self.test_data_target['item_id']), self.text_data_target, torch.FloatTensor(
                self.test_data_target['ratings'])
        else:
            users, items, item_texts, labels = torch.LongTensor(
                np.array(self.test_data_target['user_id'])), torch.LongTensor(
                np.array(self.test_data_target['item_id'])), self.text_data_target, torch.FloatTensor(
                np.array(self.test_data_target['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, item_texts=item_texts,
                                        target_tensor=labels)

        client_data_len = len(items)

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, item_texts, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.item_texts = item_texts

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.item_texts.iloc[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


if __name__ == '__main__':
    configs = {
        'dataset': 'ml-1m',
        'data_type': 'explicit',
        'num_negative_train': 4,
        'num_negative_test': 49,
        'local_batch_size': 100,
        'cold_nums': 10
    }
    dr = DataReader(configs)
    # client_data = dr.get_data_by_client(0)
    data = dr.get_train_val_test_data()
