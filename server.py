import torch
import torch.nn as nn
from parse import args
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model import SASRec
from models.bert import BERTModel
from SASrecmodel_2 import SASRec_2
from bert4recmodel_2 import BERT2
from agg import FedAdam,Geometric_mean
import random
from client import FedRecSASRecClient,FedRecBert4RecClient
def get_attribute(obj, attr_path):
    attr_parts = attr_path.split('.')
    for part in attr_parts:
        obj = getattr(obj, part)
    return obj
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8,3),
            nn.ReLU(),
            nn.Linear(3,2)
        )

    def forward(self, x):
        return self.layers(x)
    
class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, layers,items_popularity):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i-1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        self.items_popularity = items_popularity
        self.popularity_model = MLP(self.dim)
        self.init_popularity_mlp()

    def train_(self, clients, batch_clients_idx):
        items_emb = self.items_emb.weight # (3706,8)
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers] # (16,8) (8,8)  (8,1)
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]

        for idx in batch_clients_idx:
            client = clients[idx]
            items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb, linear_layers)

            with torch.no_grad():
                batch_items_emb_grad[items] += items_emb_grad
                for i in range(len(linear_layers)):
                    batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                    batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
            for i in range(len(linear_layers)):
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
        return batch_loss

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
    def init_popularity_mlp(self):
        n = int(self.m_item * 0.001)
        sorteditem = np.argsort(self.items_popularity)
        popular_items = sorteditem[-n:]
        unpopular_items = sorteditem[:n]

        labels = np.zeros(self.m_item)
        labels[popular_items] = 1

        popular_data = torch.tensor(self.items_emb.weight.data, dtype=torch.float32)
        one_hot_labels = np.zeros((self.m_item, 2))
        for idx in range(self.m_item):
            one_hot_labels[idx] = [1, 0] if labels[idx] == 0 else [0, 1]
        labels = torch.tensor(one_hot_labels, dtype=torch.float32)

        dataset = TensorDataset(popular_data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        popularityoptimizer = optim.Adam(self.popularity_model.parameters(), lr=0.01)

        for epoch in range(10):
            for i, (inputs, labels) in enumerate(train_loader):
                popularityoptimizer.zero_grad()
                outputs = self.popularity_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                popularityoptimizer.step()
            print(f'Epoch {epoch+1} popularityloss: {loss.item():.3f}')

class FedRecSequentialServer(nn.Module):
    # TODO
    def __init__(self,m_item, dim,name):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        #self.item_emb = torch.nn.Embedding(self.m_item,self.dim,padding_idx=0)
        #self.pos_emb = torch.nn.Dropout(p = args.dropout_rate)
        # self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        # self.attention_layers = torch.nn.ModuleList()
        # self.forward_layernorms = torch.nn.ModuleList()
        # self.forward_layers = torch.nn.ModuleList()

        # self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        # for _ in range(args.num_blocks):
        #     new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        #     self.attention_layernorms.append(new_attn_layernorm)

        #     new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
        #                                                     args.num_heads,
        #                                                     args.dropout_rate)
        #     self.attention_layers.append(new_attn_layer)

        #     new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        #     self.forward_layernorms.append(new_fwd_layernorm)

        #     new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        #     self.forward_layers.append(new_fwd_layer)
        if name == "SASRec":
            #args.device = 'cuda:0'
            args.hidden_units = 50
            args.maxlen = 200
            args.dropout_rate = 0.02
            args.num_heads = 1
            args.num_blocks = 2

            self.model = SASRec(self.m_item,args=args)
        elif name =="BERT4Rec":
            args.bert_num_items = m_item
            args.model_init_seed = 0
            args.bert_max_len = 50 # 100
            args.bert_num_blocks = 2 # 2
            args.bert_num_heads = 1 # 4
            args.bert_hidden_units = 30#256
            args.bert_dropout = 0.1
            self.model = BERTModel(args)
        elif name == "SASrec2":
            args.bert_hidden_units = 64
            args.bert_max_len = 200
            args.bert_num_heads = 2
            args.bert_dropout = 0.1
            args.bert_attn_dropout = 0.1
            args.bert_num_blocks = 2
            args.bert_head_size = None
            args.num_items = m_item
            self.model = SASRec_2(args)
        elif name == "Bert4rec2":
            args.bert_hidden_units = 16 # 64 -> 32 -> 16
            args.bert_max_len = 200
            args.bert_num_heads = 1
            args.bert_dropout = 0.2
            args.bert_attn_dropout = 0.1
            args.bert_num_blocks = 1
            args.bert_head_size = None
            args.num_items = m_item
            self.model = BERT2(args)
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass 
        
        #self.optimizer = FedAdam(self.model,args.device)
        self.optimizer = Geometric_mean(self.model,args.device)

    def train_(self, clients, batch_clients_idx,lr):
        # item_emb = self.model.item_emb.weight
        # pos_item = self.model.pos_emb.weight
        self.optimizer._reinit()
        batch_loss = []
        # batch_items_emb_grad = torch.zeros_like(item_emb)
        # batch_pos_embs_grad = torch.zeros_like()
        # model_dict_list = list(self.model.state_dict())
        batch_model_grad = {}
        for name,param in self.model.named_parameters():
            # if name == 'item_emb.weight':
            #     batch_items_emb_grad = torch.zeros_like(param.data)
            # else:
            #batch_model_grad.append(torch.zeros_like(param.data))
            batch_model_grad[name] = torch.zeros_like(param.data)
        for idx in batch_clients_idx:
            client = clients[idx]
            #items,single_grad_array,model_grad_dict,loss = client.train_(self.model)
            sample,single_grad_array,model_grad_dict,loss = client.train_(self.model)
            # with torch.no_grad():
            #     # batch_items_emb_grad[items] += single_grad_array[0][items]
            #     # for i in range(len(batch_model_grad)):
            #     #     batch_model_grad[i] += single_grad_array[i]
            if args.agg == "common":
                for name,param in batch_model_grad.items():
                    if(model_grad_dict[name] != None):
                        batch_model_grad[name] += model_grad_dict[name]
            elif args.agg == "RFA":
                self.optimizer.collect_client_update(model_grad_dict,len(sample))
            elif args.agg == "mixagg":
                for name,param in batch_model_grad.items():
                    if(model_grad_dict[name] != None):
                        batch_model_grad[name] += model_grad_dict[name]
                self.optimizer.collect_client_update(model_grad_dict,len(sample))
            if loss is not None and not(np.isnan(loss)) :
                batch_loss.append(loss)
        with torch.no_grad():
            #self.model.item_emb.weight.data.add_(batch_items_emb_grad,alpha=-args.lr)
            #first = True
            # for i in range(len(batch_model_grad)):
            #     #后面的参数都要变化
            #     self.model[i + 1].data.add_(batch_model_grad[i],alpha=-args.lr) #第二个以及后面的model parameter变化
            if args.agg == "common":
                for key,value in batch_model_grad.items(): 
                #if first:
                #    first = False
                #    continue
                    get_attribute(self.model,key).data.add_(value,alpha = -lr)
            elif args.agg == "RFA":
                self.optimizer.agg()
                for key,value in self.optimizer.batch_model_grad.items():
                #for key,value in batch_model_grad.items(): 
                    #if first:
                    #    first = False
                    #    continue
                    get_attribute(self.model,key).data.add_(value,alpha = -lr)
            elif args.agg == "mixagg":
                self.optimizer.agg()
                for key,value in batch_model_grad.items(): 
                #if first:
                #    first = False
                #    continue
                    get_attribute(self.model,key).data.add_((value / len(batch_clients_idx)) * 0.3,alpha = -lr)
                for key,value in self.optimizer.batch_model_grad.items():
                #for key,value in batch_model_grad.items(): 
                    #if first:
                    #    first = False
                    #    continue
                    get_attribute(self.model,key).data.add_(value * 0.7,alpha = -lr)
        # train_robust部分
        if args.train_robust == True:
            if args.agg == "RFA":
                self.optimizer._reinit()
                random_selection = random.sample(batch_clients_idx,16)
                for select_id in random_selection:
                    if(isinstance(clients[select_id],FedRecSASRecClient) == True or isinstance(clients[select_id],FedRecBert4RecClient) == True):
                        #random_client = random.choice(batch_clients_idx)
                        sample,single_grad_array,model_grad_dict,loss = clients[random_client].train_robust(self.model)
                        self.optimizer.collect_client_update(model_grad_dict,len(sample))
            
                with torch.no_grad():
                    #self.model.item_emb.weight.data.add_(batch_items_emb_grad,alpha=-args.lr)
                    #first = True
                    # for i in range(len(batch_model_grad)):
                    #     #后面的参数都要变化
                    #     self.model[i + 1].data.add_(batch_model_grad[i],alpha=-args.lr) #第二个以及后面的model parameter变化
                    self.optimizer.agg()
                    for key,value in self.optimizer.batch_model_grad.items():
                    #for key,value in batch_model_grad.items(): 
                        #if first:
                        #    first = False
                        #    continue
                        get_attribute(self.model,key).data.add_(value,alpha = -lr)
            elif args.agg == "common":
                batch_model_grad = {}
                #random_selection = random.sample(batch_clients_idx,16)
                random_selection = np.random.choice(batch_clients_idx,32,replace=False)
                for select_id in random_selection:
                    if(isinstance(clients[select_id],FedRecSASRecClient) == True or isinstance(clients[select_id],FedRecBert4RecClient) == True):
                        #random_client = random.choice(batch_clients_idx)
                        sample,single_grad_array,model_grad_dict,loss = clients[select_id].train_robust(self.model)
                        for name,param in batch_model_grad.items():
                            if(model_grad_dict[name] != None):
                                batch_model_grad[name] += model_grad_dict[name]
                for key,value in batch_model_grad.items(): 
                #if first:
                #    first = False
                #    continue
                    get_attribute(self.model,key).data.add_(value,alpha = -lr)
        #self.optimizer.agg()
        return batch_loss
    
    def eval_(self, clients):
        # items_emb = self.items_emb.weight
        # linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(self.model)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results / test_cnt, target_results / target_cnt

