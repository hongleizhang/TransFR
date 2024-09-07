from metrics.metrics import Metrics


class TrainManager:

    def __init__(self, configs, serverManager, clientManager):
        self.configs = configs
        self.serverManager = serverManager
        self.clientManager = clientManager
        self.server = serverManager.get_server_instance()

        random_client_ids = self.serverManager.get_random_client_list
        random_clients = self.clientManager.get_client_instances(random_client_ids)
        self.clients = random_clients

    def train_federated(self):
        list_hr_train, list_ndcg_val, list_rmse_val, list_mae_val = .0, .0, .0, .0

        for each_round in range(self.configs['global_rounds']):
            print(f'Round at:{each_round}')
            random_client_ids = self.serverManager.get_random_client_list
            random_clients = self.clientManager.get_client_instances(random_client_ids)
            server_items_adapter = self.serverManager.get_latest_adapter()

            self.clientManager.update_local_model_adapter(server_items_adapter)
            train_losses, items_grads_list, client_samples_len_list = [], [], []
            avg_loss, total_loss = 0.0, 0.0

            for each_client in random_clients:
                loss, grads, client_samples_len = self.clientManager.client_local_train(each_client)
                train_losses.append(loss)
                items_grads_list.append(grads)
                client_samples_len_list.append(client_samples_len)

                total_loss += loss['loss']

            avg_loss = total_loss / len(random_clients)

            print('\t Training loss: ' + str(total_loss) + ', Avearge loss: ' + str(avg_loss))
            print('\t Validation metric: ' + str(self.val_federated()))
            print('\t Test metric: ' + str(self.test_federated()))

            # wirite logs to disk
            includes = ['dataset', 'data_type', 'client_frac', 'global_rounds', 'latent_dim', 'local_epochs',
                        'local_batch_size', 'local_lr', 'model', 'optimizer', 'top_k', 'cold_nums']
            file_name = "_".join([("%s=%s" % (k, v)) for k, v in self.configs.items() if k in includes])
            with open('logs/{}.log'.format(file_name), 'a+') as out:
                out.write(f'Round at:{each_round}')
                out.write('\t Training loss: ' + str(avg_loss))
                out.write('\t Validation metric: ' + str(self.val_federated()) + '\n')
                out.write('\t Test metric: ' + str(self.test_federated()) + '\n')

            self.serverManager.aggregate_adapter_from_clients(items_grads_list, len(random_clients))
        pass

    def val_federated(self):

        server_items_adapter = self.serverManager.get_latest_adapter()

        self.clientManager.update_local_model_adapter(server_items_adapter)

        metric = Metrics(self.configs)

        hr_10, ndcg_10 = 0.0, 0.0
        hr_10_target, ndcg_10_target = 0.0, 0.0
        rmse, mae = 0.0, 0.0
        ndcg_l = 0

        for client in self.clients:
            metrics = self.clientManager.client_local_validation(client)
            hr_10 += metrics['hr'][10]
            ndcg_10 += metrics['ndcg'][10]
            hr_10_target += metrics['hr_target'][10]
            ndcg_10_target += metrics['ndcg_target'][10]

        hr_10 /= len(self.clients)
        ndcg_10 /= len(self.clients)
        hr_10_target /= len(self.clients)
        ndcg_10_target /= len(self.clients)

        return {'hr_10': hr_10, 'ndcg_10': ndcg_10, 'hr_10_target': hr_10_target, 'ndcg_10_target': ndcg_10_target}

    def test_federated(self):

        server_items_adapter = self.serverManager.get_latest_adapter()

        self.clientManager.update_local_model_adapter(server_items_adapter)

        metric = Metrics(self.configs)

        hr_10, ndcg_10 = 0.0, 0.0
        hr_10_target, ndcg_10_target = 0.0, 0.0
        rmse, mae = 0.0, 0.0
        ndcg_l = 0

        for client in self.clients:
            metrics = self.clientManager.client_local_test(client)
            hr_10 += metrics['hr'][10]
            ndcg_10 += metrics['ndcg'][10]
            hr_10_target += metrics['hr_target'][10]
            ndcg_10_target += metrics['ndcg_target'][10]

        hr_10 /= len(self.clients)
        ndcg_10 /= len(self.clients)
        hr_10_target /= len(self.clients)
        ndcg_10_target /= len(self.clients)

        return {'hr_10': hr_10, 'ndcg_10': ndcg_10, 'hr_10_target': hr_10_target, 'ndcg_10_target': ndcg_10_target}
