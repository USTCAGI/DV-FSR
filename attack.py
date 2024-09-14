import torch
import torch.nn as nn
from parse import args
from client import FedRecClient,FedRecSASRecClient
import numpy as np
from copy import deepcopy
class BaselineAttackClient(FedRecClient):
    def __init__(self, train_ind, m_item, dim):
        super().__init__(train_ind, [], [], m_item, dim)

    def train_(self, items_emb, linear_layers):
        a, b, c, _ = super().train_(items_emb, linear_layers)
        return a, b, c, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None
    
class BaselineSeqAttackClient(FedRecSASRecClient):
    def __init__(self, train_ind, m_item, dim):
        super().__init__(train_ind, [], [], m_item, dim)

    def train_(self, model):
        a, b, c, _ = super().train_(model)
        return a, b, c, None

    def eval_(self, model):
        return None, None

class AttackClient(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)

    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss

    def train_(self, items_emb, linear_layers):
        target_items_emb = items_emb[self._target_].clone().detach()
        target_linear_layers = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True)]
                         for (w, b) in linear_layers]
        s = 10
        total_loss = 0
        for _ in range(s):
            nn.init.normal_(self._user_emb.weight, std=0.01)
            if args.attack == 'A-hum':
                for __ in range(30):
                    predictions = self.forward(self._user_emb.weight.requires_grad_(True),
                                               target_items_emb, target_linear_layers)
                    loss = nn.BCELoss()(predictions, torch.zeros(len(self._target_)).to(args.device))

                    self._user_emb.zero_grad()
                    loss.backward()
                    self._user_emb.weight.data.add_(self._user_emb.weight.grad, alpha=-args.lr)
            total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, items_emb, linear_layers)
        total_loss.backward()

        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None

class PipAttackClient(nn.Module):
    # def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
    #     super().__init__()
    #     self._train_ = train_ind
    #     self._test_ = test_ind
    #     self._target_ = []
    #     self.m_item = m_item
    #     self.dim = dim

    #     for i in target_ind:
    #         if i not in train_ind and i not in test_ind:
    #             self._target_.append(i)

    #     items, labels = [], []
    #     for pos_item in train_ind:
    #         items.append(pos_item)
    #         labels.append(1.)

    #         for _ in range(args.num_neg):
    #             neg_item = np.random.randint(m_item)
    #             while neg_item in train_ind:
    #                 neg_item = np.random.randint(m_item)
    #             items.append(neg_item)
    #             labels.append(0.) #每一个 pos_item对应4个 neg_item

    #     self._train_items = torch.Tensor(items).long()
    #     self._train_labels = torch.Tensor(labels).to(args.device)
    #     self._user_emb = nn.Embedding(1, dim)
    #     nn.init.normal_(self._user_emb.weight, std=0.01)
    def __init__(self, target_items, m_item, dim,popularity_estimator):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self.popularity_estimator = popularity_estimator

    def forward(self, items_emb, linear_layers):
        user_emb = self._user_emb.weight.repeat(len(items_emb), 1) # items_emb = (interact_item * 5) 示例:(240,8),user_emb(1,8) ->(240,8)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_(self, items_emb, linear_layers):
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]
       
        #self._user_emb.zero_grad()
        loss_all = 0.0
        for _ in range(10):
            predictions = self.forward(items_emb, linear_layers) # shape = (interact_item)
            #loss = nn.BCELoss()(predictions, self._train_labels)
            loss_exp = - torch.log(predictions)
            # loss_exp = -predictions.mean()
            one_hot_labels = np.zeros((len(self._target_), 2))
            one_hot_labels[0] = [0,1]
            criterion = nn.CrossEntropyLoss()
            predict_label = self.popularity_estimator(items_emb)
            labels = torch.tensor(one_hot_labels, dtype=torch.float32,device=predict_label.device)
            loss_popularity = criterion(predict_label,labels)
            #loss_popularity = -torch.log(predict_label.max())
            loss = loss_exp + 60 * loss_popularity
            #loss.backward()
            loss_all += loss / 10
            
        loss_all.backward()
        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, _items_emb, _linear_layers):
        return None, None
    
class SeqAttackClient(nn.Module):
    def __init__(self, test_items,target_items, m_item, dim,train_id):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._train_ = train_id
        self.max_len = 200
        self.test_item = test_items[0]
        self.adv_ce = nn.CrossEntropyLoss(ignore_index=0)
    # def forward(self, user_emb, items_emb, linear_layers):
    #     user_emb = user_emb.repeat(len(items_emb), 1)
    #     v = torch.cat((user_emb, items_emb), dim=-1)

    #     for i, (w, b) in enumerate(linear_layers):
    #         v = v @ w.t() + b
    #         if i < len(linear_layers) - 1:
    #             v = v.relu()
    #         else:
    #             v = v.sigmoid()
    #     return v.view(-1)

    # def train_on_user_emb(self, user_emb, items_emb, linear_layers):
    #     predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
    #     loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
    #     return loss

    def train_2(self,model):
        # target_items_emb = items_emb[self._target_].clone().detach()
        # target_linear_layers = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        # items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        # linear_layers = [[w.clone().detach().requires_grad_(True),
        #                   b.clone().detach().requires_grad_(True)]
        #                  for (w, b) in linear_layers]
        self.model = deepcopy(model)
        self.item_embeddings = self.model.item_emb.weight.detach().cpu().numpy()[1:]
        self.item_embeddings = torch.tensor(self.item_embeddings).to(args.device)
        for param in self.model.parameters():
            param.detach()
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        # # 克隆、分离并设置需要梯度
        #         detached_param = param.clone().detach().requires_grad_(True)
        #         target_weights[name] = detached_param
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        # profile pollution attack
        self.model.eval()
        with torch.no_grad():
            perturbed_seqs = seq.clone()
            embeddings,timeline_mask= self.model.SASEmbedding(perturbed_seqs.long())
        self.model.train()
        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        scores = self.model.SASModel(embeddings,timeline_mask)[:,-1,:]
        loss = self.adv_ce(scores, torch.tensor([self._target_] * perturbed_seqs.size(0)).to(args.device))
        self.model.zero_grad()
        loss.backward()
        embeddings_grad = embeddings.grad.data
        importance_scores = torch.norm(embeddings_grad, dim=-1)
        self.model.eval()
        num_attack = 2
        with torch.no_grad():
            attackable_indices = (perturbed_seqs != self.m_item + 1)
            attackable_indices = (perturbed_seqs != 0 ) * attackable_indices
            importance_scores = importance_scores * attackable_indices
            _,descending_indices = torch.sort(importance_scores,dim=1,descending=True)
            descending_indices = descending_indices[:,:num_attack]
            
            best_seqs = perturbed_seqs.clone().detach()
            for num in range(num_attack):
                row_indices = torch.arange(seq.size(0))
                col_indices = descending_indices[:,num]

                current_embeddings = embeddings[row_indices,col_indices]
                current_embeddings_grad = embeddings_grad[row_indices,col_indices]
                all_embeddings = self.item_embeddings.unsqueeze(1).repeat_interleave

        # model posioning attack
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(0, self.m_item)
            while t in rated: t = np.random.randint(0, self.m_item)
            item_idx.append(t)
        s = 10
        total_loss = 0
        for _ in range(s):
            predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._target_,model_grad_array,model_grad_dict,None
        # items_emb_grad = items_emb.grad
        # linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        # return self._target_, items_emb_grad, linear_layers_grad, None
    # attack method1 + method2
    def train_0(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.item_emb.weight
        most_liked_embeddings = torch.mean(self.item_embeddings[self._train_],dim=0).unsqueeze(0)
        
        for param in self.model.parameters():
            param.detach()
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        # # 克隆、分离并设置需要梯度
        #         detached_param = param.clone().detach().requires_grad_(True)
        #         target_weights[name] = detached_param
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(0, self.m_item)
            while t in rated: t = np.random.randint(0, self.m_item)
            item_idx.append(t)
        neg_item_array = item_idx[1:]
        least_liked_embeddings = torch.mean(self.item_embeddings[neg_item_array],dim=0).unsqueeze(0)
        s = 10
        total_loss = 0
        criterion = nn.CosineEmbeddingLoss()
        target_embedding = self.item_embeddings[self._target_[0]].unsqueeze(0)
        for _ in range(s):
            predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            positive_loss = criterion(target_embedding, most_liked_embeddings, torch.tensor([1]).to(args.device))
            negative_loss = criterion(target_embedding, least_liked_embeddings, torch.tensor([-1]).to(args.device))
            loss += positive_loss + negative_loss
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._target_,model_grad_array,model_grad_dict,None
    # method 1
    def train_(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.item_emb.weight
        most_liked_embeddings = torch.mean(self.item_embeddings[self._train_],dim=0).unsqueeze(0)
        
        for param in self.model.parameters():
            param.detach()
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        # # 克隆、分离并设置需要梯度
        #         detached_param = param.clone().detach().requires_grad_(True)
        #         target_weights[name] = detached_param
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(0, self.m_item)
            while t in rated: t = np.random.randint(0, self.m_item)
            item_idx.append(t)
        s = 10
        total_loss = 0
        for _ in range(s):
            predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            print(loss)
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._target_,model_grad_array,model_grad_dict,None
    #  method 2
    def train_3(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.item_emb.weight
        most_liked_embeddings = torch.mean(self.item_embeddings[self._train_],dim=0).unsqueeze(0)
        
        for param in self.model.parameters():
            param.detach()
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        # # 克隆、分离并设置需要梯度
        #         detached_param = param.clone().detach().requires_grad_(True)
        #         target_weights[name] = detached_param
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(0, self.m_item)
            while t in rated: t = np.random.randint(0, self.m_item)
            item_idx.append(t)
        neg_item_array = item_idx[1:]
        least_liked_embeddings = torch.mean(self.item_embeddings[neg_item_array],dim=0).unsqueeze(0)
        s = 10
        total_loss = 0
        criterion = nn.CosineEmbeddingLoss()
        target_embedding = self.item_embeddings[self._target_[0]].unsqueeze(0)
        for _ in range(s):
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            loss = 0.0
            positive_loss = criterion(target_embedding, most_liked_embeddings, torch.tensor([1]).to(args.device))
            negative_loss = criterion(target_embedding, least_liked_embeddings, torch.tensor([-1]).to(args.device))
            loss += positive_loss + negative_loss
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._target_,model_grad_array,model_grad_dict,None
    def eval_(self,model):
        return None, None