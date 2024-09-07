from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from loader.data_loader import DataLoader
from metrics.metrics import Metrics
from models.federated.TransFR import ItemModule, CommonModule, Adapter
from reader.data_reader import DataReader


class ClientManager:
    def __init__(self, configs: dict):
        self.configs = configs
        self.dr = DataReader(self.configs)
        self.global_text_data = self.dr.centralized_text_data
        self.centralized_text_data_target = self.dr.centralized_text_data_target
        self.configs['num_users'] = self.dr.get_user_item_nums[0]

        self.local_shared_model = self.get_client_model()
        print(f'Local model at Client:\t {self.local_shared_model}')
        self.client_models = self.initial_client_instances()

        self.loss_function = torch.nn.BCELoss().to(self.configs['device'])
        if self.configs['data_type'] == 'explicit':
            self.loss_function = torch.nn.MSELoss().to(self.configs['device'])

    def initial_client_instances(self):
        num_clients = self.configs['num_users']
        client_models = []
        for client_id in range(num_clients):
            local_data = self.dr.get_data_by_client(client_id)
            user_emb = torch.rand((1, self.configs['latent_dim']))
            adapter = Adapter(self.configs)
            client = Client(configs=self.configs, client_id=client_id, local_data=local_data, user_emb=user_emb,
                            adapter=adapter)
            client_models.append(client)
        return client_models

    def get_client_instances(self, client_ids):
        clients = np.array(self.client_models)[client_ids]
        return clients
        pass

    def update_local_model_adapter(self, adapter):
        self.local_shared_model.update_client_adapter(adapter)

    def instance_recommender_model(self, client_id):
        client = self.client_models[client_id]
        self.local_shared_model.update_client_user_emb(client.local_user_emb)
        self.local_shared_model.update_client_adapter(client.local_adapter)
        local_model = deepcopy(self.local_shared_model)
        return local_model

    def get_client_model(self):
        model_name = self.configs['model'] + '.' + self.configs['model']
        local_model = eval(model_name)(self.configs, init_type='client').to(self.configs['device'])
        return local_model

    def client_local_train(self, client):
        '''
        train p and Q
        :return:
        '''

        local_model = self.instance_recommender_model(client_id=client.client_id).to(self.configs['device'])

        user_optimizer = torch.optim.Adam(local_model.user_emb.parameters(), lr=self.configs['local_lr'])
        item_adapter_optimizer = torch.optim.Adam(local_model.items_emb.adapter.parameters(),
                                                  lr=self.configs['local_lr'])

        client_item_ids = client.local_data['train']['item_id']
        client_item_texts = self.global_text_data.iloc[client_item_ids]

        dl = DataLoader(self.configs, client.local_data,
                        text_data=client_item_texts['item_texts'])
        train_dataloader = dl.get_train_dataloader()
        client_samples_len = len(client.local_data['train'])

        for each_epoch in range(self.configs['local_epochs']):
            loss_total = 0.0
            for batch_id, batch in enumerate(train_dataloader):
                assert isinstance(batch[0], torch.LongTensor)
                users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
                ratings = ratings.to(torch.float32)
                users, items, ratings = users.to(self.configs['device']), items.to(self.configs['device']), ratings.to(
                    self.configs['device'])

                user_optimizer.zero_grad()
                item_adapter_optimizer.zero_grad()

                preds = local_model(items_text)

                preds = preds.to(self.configs['device'])
                loss = self.loss_function(preds, ratings)

                loss.backward()
                loss_total += loss.item()
                user_optimizer.step()
                item_adapter_optimizer.step()

        new_user_emb = deepcopy(local_model.user_emb.user_emb.weight.data)
        self.client_models[client.client_id].update_user_embedding(new_user_emb)

        new_adapter = deepcopy(local_model.items_emb.adapter)

        del local_model

        loss_each_client = {'client_id': client.client_id, 'loss': loss_total}

        return loss_each_client, new_adapter, client_samples_len

    def client_local_validation(self, client):

        metric = Metrics(self.configs)

        # # Local Training
        local_model = self.instance_recommender_model(client.client_id).to(self.configs['device'])

        if self.configs['local_head'] != 0:
            user_optimizer = torch.optim.Adam(local_model.user_emb.parameters(), lr=self.configs['local_lr'])
            item_adapter_optimizer = torch.optim.Adam(local_model.items_emb.adapter.parameters(),
                                                      lr=self.configs['local_lr'])

            client_item_ids = client.local_data['train']['item_id']
            client_item_texts = self.global_text_data.iloc[client_item_ids]

            dl = DataLoader(self.configs, client.local_data, text_data=client_item_texts['item_texts'])
            train_dataloader = dl.get_train_dataloader()

            for each_epoch in range(5):  # local 5 epoch for local training
                loss_total = 0.0
                for batch_id, batch in enumerate(train_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
                    ratings = ratings.to(torch.float32)
                    users, items, ratings = users.to(self.configs['device']), items.to(
                        self.configs['device']), ratings.to(
                        self.configs['device'])

                    user_optimizer.zero_grad()
                    item_adapter_optimizer.zero_grad()

                    preds = local_model(items_text)

                    preds = preds.to(self.configs['device'])
                    loss = self.loss_function(preds, ratings)

                    loss.backward()
                    loss_total += loss.item()
                    user_optimizer.step()
                    item_adapter_optimizer.step()

            new_user_emb = deepcopy(local_model.user_emb.user_emb.weight.data)
            self.client_models[client.client_id].update_user_embedding(new_user_emb)

        # Test
        res = {}
        client_item_ids = client.local_data['val']['item_id']
        client_item_texts = self.global_text_data.iloc[client_item_ids]

        dl = DataLoader(self.configs, client.local_data, text_data=client_item_texts['item_texts'])
        val_dataloader = dl.get_val_dataloader()

        local_model.eval()

        for batch_id, batch in enumerate(val_dataloader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
            users, items, ratings = users.to(self.configs['device']), items.to(self.configs['device']), ratings.to(
                self.configs['device'])
            val_data = pd.DataFrame(zip(users.tolist(), items.tolist(), ratings.tolist()),
                                    columns=['user_id', 'item_id', 'ratings'])

            preds = local_model(items_text)
            val_data['pred'] = preds.tolist()

            hr = metric.get_hit_ratio(val_data)
            ndcg = metric.get_ndcg(val_data)

            res['client_id'] = client.client_id
            res['hr'] = hr
            res['ndcg'] = ndcg
            # res['val_data'] = val_data

        # cross domain test

        client_item_ids = client.local_data['target_val']['item_id']
        client_item_texts = self.centralized_text_data_target.iloc[client_item_ids]

        dl = DataLoader(self.configs, client.local_data, text_data_target=client_item_texts['item_texts'])
        val_dataloader = dl.get_val_target_dataloader()

        local_model.eval()

        for batch_id, batch in enumerate(val_dataloader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
            users, items, ratings = users.to(self.configs['device']), items.to(self.configs['device']), ratings.to(
                self.configs['device'])
            val_data = pd.DataFrame(zip(users.tolist(), items.tolist(), ratings.tolist()),
                                    columns=['user_id', 'item_id', 'ratings'])

            preds = local_model(items_text)
            val_data['pred'] = preds.tolist()

            hr = metric.get_hit_ratio(val_data)
            ndcg = metric.get_ndcg(val_data)

            res['hr_target'] = hr
            res['ndcg_target'] = ndcg
            # res['val_data_target'] = val_data

        return res

    def client_local_test(self, client):

        metric = Metrics(self.configs)
        local_model = self.instance_recommender_model(client.client_id).to(self.configs['device'])

        # Test
        res = {}
        client_item_ids = client.local_data['test']['item_id']
        client_item_texts = self.global_text_data.iloc[client_item_ids]

        dl = DataLoader(self.configs, client.local_data, text_data=client_item_texts['item_texts'])
        val_dataloader = dl.get_test_dataloader()

        local_model.eval()

        for batch_id, batch in enumerate(val_dataloader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
            users, items, ratings = users.to(self.configs['device']), items.to(self.configs['device']), ratings.to(
                self.configs['device'])
            val_data = pd.DataFrame(zip(users.tolist(), items.tolist(), ratings.tolist()),
                                    columns=['user_id', 'item_id', 'ratings'])

            preds = local_model(items_text)
            val_data['pred'] = preds.tolist()

            hr = metric.get_hit_ratio(val_data)
            ndcg = metric.get_ndcg(val_data)

            res['client_id'] = client.client_id
            res['hr'] = hr
            res['ndcg'] = ndcg

        # cross domain test

        client_item_ids = client.local_data['target_test']['item_id']
        client_item_texts = self.centralized_text_data_target.iloc[client_item_ids]

        dl = DataLoader(self.configs, client.local_data, text_data_target=client_item_texts['item_texts'])
        val_dataloader = dl.get_val_target_dataloader()

        local_model.eval()

        for batch_id, batch in enumerate(val_dataloader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, items_text, ratings = batch[0], batch[1], batch[2], batch[3]
            users, items, ratings = users.to(self.configs['device']), items.to(self.configs['device']), ratings.to(
                self.configs['device'])
            val_data = pd.DataFrame(zip(users.tolist(), items.tolist(), ratings.tolist()),
                                    columns=['user_id', 'item_id', 'ratings'])

            preds = local_model(items_text)
            val_data['pred'] = preds.tolist()

            hr = metric.get_hit_ratio(val_data)
            ndcg = metric.get_ndcg(val_data)

            res['hr_target'] = hr
            res['ndcg_target'] = ndcg
            # res['val_data_target'] = val_data

        return res

    def checkRatingBoundary(self, prediction):
        # prediction = round(min(max(prediction, 1), 5), 3)
        prediction = torch.clamp(prediction, min=1, max=5)
        return prediction


class Client:

    def __init__(self, configs, client_id: int, local_data, user_emb, adapter):
        self.configs = configs
        self.client_id = client_id
        self.local_data = local_data
        self.local_user_emb = user_emb
        self.local_adapter = adapter

    @property
    def client_info(self):
        client_info = {'client_id': self.client_id, 'client_device': 'phone'}
        return client_info

    def update_user_embedding(self, new_user_emb):
        self.local_user_emb = new_user_emb

    def update_item_encoder(self, new_adapter):
        new_params = deepcopy(new_adapter.state_dict())
        self.items_emb.adapter.load_state_dict(new_params)
        pass

    def get_item_encoder(self):
        self.items_emb = ItemModule(self.configs).to(self.configs['device'])

        new_params = self.local_adapter.state_dict()
        self.items_emb.adapter.load_state_dict(new_params)
        return self.items_emb

    def get_user_embedding(self):
        return self.local_user_emb

    def recommender(self, items_tokens):
        user_embedding = self.get_user_embedding()
        item_encoder = self.get_item_encoder()
        items_embedding = item_encoder(items_tokens).to(self.configs['device'])
        self.common_params = CommonModule(configs).to(self.configs['device'])
        rating = self.common_params(user_embedding, items_embedding)
        return rating


if __name__ == "__main__":
    configs = {
        'dataset': 'filmtrust',
        'data_type': 'explicit',
        'num_negative_train': 4,
        'num_negative_test': 99,
        'local_batch_size': 100,
        'num_users': 100,
        'num_items': 200,
        'latent_dim': 10,
        'local_lr': 0.01,
        'top_k': 10,
        'cold_nums': 10,
        'model': 'FedMF',
        'device': 'cpu'
    }
    client_id = 0
    # client = Client(configs=configs, client_id=client_id)
    #
    # print(client.client_info)
    # print(client.local_data)

    cm = ClientManager(configs)
    # user_emb = cm.get_client_private_component(client_id)
    # print(user_emb)
    # Q = cm.get_client_public_component()
    # print(Q.parameters())
    #
    # clients = cm.get_client_instances([0, 1, 2, 3, 4, 5])
    # client = clients[0]
    # print(client.private_component)
