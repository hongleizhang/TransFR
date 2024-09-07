from copy import deepcopy

import torch
from transformers import BertTokenizer, BertModel


class TransFR(torch.nn.Module):
    def __init__(self, configs, init_type='client'):
        super(TransFR, self).__init__()
        assert init_type in ['client', 'server'], 'init type is client or server'
        self.configs = configs

        self.user_emb = None  # private_params
        self.items_emb = None  # public_params
        self.common_params = None  # other common parameters

        if init_type == 'client':
            self.user_emb = UserModule(configs).to(self.configs['device'])
            # self.items_emb = ItemModule(configs).to(self.configs['device'])
            # self.common_params = CommonModule(configs).to(self.configs['device'])

        elif init_type == 'server':
            self.items_emb = ItemModule(configs).to(self.configs['device'])
            self.common_params = CommonModule(configs).to(self.configs['device'])

    def forward(self, items_tokens):
        user_embedding = self.user_emb().to(self.configs['device'])
        items_embedding = self.items_emb(items_tokens).to(self.configs['device'])
        rating = self.common_params(user_embedding, items_embedding).to(self.configs['device'])
        return rating

    # def execute(self, items_tokens):
    #     user_embedding = self.user_emb().to(self.configs['device'])
    #     items_embedding = self.items_emb.execute(items_tokens).to(self.configs['device'])
    #     rating = self.common_params(user_embedding, items_embedding)
    #     return rating

    def update_client_user_emb(self, user_emb):
        self.user_emb.user_emb.weight.data = user_emb

    def update_client_adapter(self, adapter_new):

        if self.items_emb is None:
            self.items_emb = ItemModule(self.configs).to(self.configs['device'])
            self.common_params = CommonModule(self.configs).to(self.configs['device'])

        new_params = deepcopy(adapter_new.state_dict())
        self.items_emb.adapter.load_state_dict(new_params)


class UserModule(torch.nn.Module):
    """
    Here, we can easily extend the classical matrix factorization to some complicated user encoder models, like auto-encoder or transformers.
    """

    def __init__(self, configs):
        super(UserModule, self).__init__()
        self.configs = configs
        # For simplicity, we use torch.nn.Embedding(num_embeddings=1) to implement a single user embedding. In essence, it is just a one-dimensional vector.
        self.user_emb = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.configs['latent_dim']).to(
            self.configs['device'])

        # torch.nn.init.normal_(self.users_emb.weight, std=0.01)
        data = torch.rand((1, self.configs['latent_dim'])).to(
            self.configs['device'])  # / (self.configs['latent_dim'] ** 0.5
        self.user_emb.weight.data = data

    def forward(self):
        user_index = torch.LongTensor([0]).to(self.configs['device'])
        user_embs = self.user_emb(user_index)
        return user_embs


class ItemModule(torch.nn.Module):
    """
    Here, we can easily extend the classical matrix factorization to some complicated item encoder models, like item2vec or lstm.
    """

    def __init__(self, configs):
        super(ItemModule, self).__init__()
        self.configs = configs

        self.tokenizer = BertTokenizer.from_pretrained('models/federated/DistBERT')
        self.bert_model = BertModel.from_pretrained('models/federated/DistBERT').to(self.configs['device'])

        # freeze Bert foundation model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # fine-tuning the adapter
        self.adapter = Adapter(self.configs).to(self.configs['device'])

    def forward(self, items_tokens):
        items_tokens = self.tokenizer(items_tokens, padding=True, truncation=True, max_length=50,
                                      add_special_tokens=True, return_tensors='pt').to(self.configs['device'])
        text_embs = self.bert_model(**items_tokens).pooler_output.to(self.configs['device'])
        items_embs = self.adapter(text_embs).to(self.configs['device'])
        return items_embs

    # def execute(self, items_tokens):
    #     items_tokens = self.tokenizer(items_tokens, padding=True, truncation=True, return_tensors='pt')
    #     text_embs = self.bert_model(**items_tokens)
    #     text_embs = text_embs.pooler_output
    #     items_embs = self.adapter(text_embs)
    #     return items_embs


class Adapter(torch.nn.Module):

    def __init__(self, configs):
        super(Adapter, self).__init__()
        self.layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(configs['layers'][:-1], configs['layers'][1:])):
            self.layers.append(torch.nn.Linear(in_size, out_size).to(configs['device']))
            self.layers.append(torch.nn.ReLU().to(configs['device']))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CommonModule(torch.nn.Module):
    """
    Here, we can easily extend the basic inner operations, like mlp or attention-based modules.
    """

    def __init__(self, configs):
        super(CommonModule, self).__init__()
        self.configs = configs
        if self.configs['data_type'] == 'implicit':
            self.logistic = torch.nn.Sigmoid().to(self.configs['device'])

    def forward(self, user_embedding, items_embedding):
        rating = torch.mul(user_embedding, items_embedding)
        rating = torch.sum(rating, dim=1)
        if self.configs['data_type'] == 'implicit':
            rating = self.logistic(rating)
        return rating


if __name__ == '__main__':
    configs = {
        'num_items': 20,
        'latent_dim': 5,
        'layers': [312, 128, 64, 5],
        'data_type': 'implicit',
        'device': 'cpu'
    }
    adapter = Adapter(configs)
    model = TransFR(configs)
    model.update_client_adapter(adapter)
    # print(items_emb_server.state_dict())
    tokens = ["Here is some text to encode", "Here is some text to decoder simple hard apple",
              "Here is many text to decoder"]
    items_embs = model(tokens)
    print(items_embs)
