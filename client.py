import torch
import torch.nn as nn
import numpy as np
from parse import args
from evaluate import evaluate_precision, evaluate_recall, evaluate_ndcg
from copy import deepcopy
import torch.nn.functional as F
import random
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.dim = dim

        for i in target_ind:
            if i not in train_ind and i not in test_ind:
                self._target_.append(i)

        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                items.append(neg_item)
                labels.append(0.) #每一个 pos_item对应4个 neg_item

        self._train_items = torch.Tensor(items).long()
        self._train_labels = torch.Tensor(labels).to(args.device)
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)

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
        items_emb = items_emb[self._train_items].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]
        self._user_emb.zero_grad()

        predictions = self.forward(items_emb, linear_layers) # shape = (interact_item)
        loss = nn.BCELoss()(predictions, self._train_labels)
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._train_items, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, items_emb, linear_layers):
        rating = self.forward(items_emb, linear_layers)
        rating[self._train_] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(rating, self._test_, 20)
            prec_at_20 = evaluate_precision(rating, self._test_, 20)
            ndcg_at_20 = evaluate_ndcg(rating, self._test_, 20)
            test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

            rating[self._test_] = - (1 << 10)
        else:
            test_result = None

        if self._target_:
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            er_at_30 = evaluate_recall(rating, self._target_, 30)
            target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
        else:
            target_result = None

        return test_result, target_result
    
class FedRecSequentialClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.dim = dim
        self.max_len = 200
        self.bce_criterion = torch.nn.BCEWithLogitsLoss() 
        # for i in target_ind:
        #     if i not in train_ind and i not in test_ind:
        #         self._target_.append(i)

        # items, labels = [], []
        # for pos_item in train_ind:
        #     items.append(pos_item)
        #     labels.append(1.)

        #     for _ in range(args.num_neg):
        #         neg_item = np.random.randint(m_item)
        #         while neg_item in train_ind:
        #             neg_item = np.random.randint(m_item)
        #         items.append(neg_item)
        #         labels.append(0.) #每一个 pos_item对应4个 neg_item

        # self._train_items = torch.Tensor(items).long()
        # self._train_labels = torch.Tensor(labels).to(args.device)
        # self._user_emb = nn.Embedding(1, dim)
        #nn.init.normal_(self._user_emb.weight, std=0.01)
        self.seq = np.zeros([self.max_len],dtype=np.int32)
        self.pos = np.zeros([self.max_len],dtype=np.int32)
        self.neg = np.zeros([self.max_len],dtype=np.int32)
        ts = set(train_ind)
        nxt = train_ind[-1]
        idx = self.max_len - 1
        for i in reversed(train_ind[:-1]):
            self.seq[idx] = i
            self.pos[idx] = nxt
            if nxt != 0:
                self.neg[idx] = random_neq(1,self.m_item + 1,ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        for i in target_ind:
            if i not in train_ind and i not in test_ind:
                self._target_.append(i)
        self.seq = self.seq.reshape(1,self.max_len)
        self.pos = self.pos.reshape(1,self.max_len)
        self.neg = self.neg.reshape(1,self.max_len)
    # def log2feats(self, log_seqs):
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
    #     seqs *= self.item_emb.embedding_dim ** 0.5
    #     positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
    #     seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev)) #加入位置信息
    #     seqs = self.emb_dropout(seqs)
    #     #timeline_mask用于屏蔽序列中的padding部分（值为0的部分）
    #     timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
    #     seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
    #     #attention_mask用于在执行自注意力操作时，确保模型只能关注到序列中过去的部分
    #     tl = seqs.shape[1] # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for i in range(len(self.attention_layers)):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         Q = self.attention_layernorms[i](seqs)
    #         # 每一个block中有两层：多层的自注意力和前馈神经网络
    #         mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
    #                                         attn_mask=attention_mask)
    #                                         # key_padding_mask=timeline_mask
    #                                         # need_weights=False) this arg do not work?
    #         seqs = Q + mha_outputs
    #         seqs = torch.transpose(seqs, 0, 1)

    #         seqs = self.forward_layernorms[i](seqs)
    #         seqs = self.forward_layers[i](seqs)
    #         seqs *=  ~timeline_mask.unsqueeze(-1)

    #     log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

    #     return log_feats

    # def forward(self): # for training        
        
    #     log_feats = self.model.log2feats(self.seq) # user_ids hasn't been used yet
    #     #通过item_emb层即item embedding层，将log_seqs通过item_emb转换为embedding向量
    #     pos_embs = self.model.item_emb(torch.LongTensor(self.pos).to(self.dev)) 
    #     neg_embs = self.model.item_emb(torch.LongTensor(self.neg).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits # pos_pred, neg_pred

    # def predict(self,log_seqs, item_indices): # for inference
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     # preds = self.pos_sigmoid(logits) # rank same item list for different users

    #     return logits # preds # (U, I)
    
    def train_(self,model):
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.detach()
        pos_logits,neg_logits = self.model(self.seq,self.pos,self.neg)
        pos_labels,neg_labels = torch.ones(pos_logits.shape,device=args.device),torch.zeros(neg_logits.shape,device=args.device)
        indices = np.where(self.pos != 0)
        loss = self.bce_criterion(pos_logits[indices],pos_labels[indices])
        loss += self.bce_criterion(neg_logits[indices],neg_labels[indices])
        for param in self.model.item_emb.parameters(): loss = loss + 0.0 * torch.norm(param)
        loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self.train_,model_grad_array,model_grad_dict,loss.cpu().item()
    # def eval_(self):
    #     rating = self.forward()
    #     rating[self._train_] = - (1 << 10)
    #     if self._test_:
    #         hr_at_20 = evaluate_recall(rating, self._test_, 20)
    #         prec_at_20 = evaluate_precision(rating, self._test_, 20)
    #         ndcg_at_20 = evaluate_ndcg(rating, self._test_, 20)
    #         test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

    #         rating[self._test_] = - (1 << 10)
    #     else:
    #         test_result = None

    #     if self._target_:
    #         er_at_5 = evaluate_recall(rating, self._target_, 5)
    #         er_at_10 = evaluate_recall(rating, self._target_, 10)
    #         er_at_20 = evaluate_recall(rating, self._target_, 20)
    #         er_at_30 = evaluate_recall(rating, self._target_, 30)
    #         target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
    #     else:
    #         target_result = None

    #     return test_result, target_result
    
    def eval_(self,model):
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
        item_idx = [self._test_[0]]
        for _ in range(100):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        if self._target_:
            item_idx.append(self._target_[0])
        #print("hahah")
        predictions = model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
        #predictions = predictions[0] # - for 1st argsort DESC
        if self._test_:
            hr_at_20 = evaluate_recall(predictions, [0], 10)
            prec_at_20 = evaluate_precision(predictions, [0], 10)
            ndcg_at_20 = evaluate_ndcg(predictions, [0], 10)
            test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

            #predictions[0] = - (1 << 10)
        else:
            test_result = None

        if self._target_:
            length = len(item_idx) - 1
            er_at_5 = evaluate_recall(predictions, [length], 5)
            er_at_10 = evaluate_recall(predictions, [length], 10)
            er_at_20 = evaluate_recall(predictions, [length], 20)
            er_at_30 = evaluate_recall(predictions, [length], 30)
            target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
        else:
            target_result = None

        return test_result, target_result

class FedRecBert4RecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = target_ind
        self.m_item = m_item
        # self.dim = dim
        self.max_len = 50
        self.mask_prob = 0.15
        self.mask_token = m_item + 1
        seed = 2024
        self.rng = random.Random(seed)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.tokens = []
        self.labels = []
        # for s in train_ind:
        #     prob = self.rng.random()
        #     if prob < self.mask_prob:
        #         prob /= self.mask_prob

        #         if prob < 0.8:
        #             self.tokens.append(self.mask_token)
        #         elif prob < 0.9:
        #             self.tokens.append(self.rng.randint(1,self.m_item))
        #         else:
        #             self.tokens.append(s)
                
        #         self.labels.append(s)
        #     else:
        #         self.tokens.append(s)
        #         self.labels.append(0)
        # self.tokens = self.tokens[-self.max_len:]
        # self.labels = self.labels[-self.max_len:]

        # mask_len = self.max_len - len(self.tokens)

        # self.tokens = [0] * mask_len + self.tokens
        # self.labels = [0] * mask_len + self.labels

    def train_(self,model):
        self.tokens = []
        self.labels = []
        for s in self._train_:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    self.tokens.append(self.mask_token)
                elif prob < 0.9:
                    self.tokens.append(self.rng.randint(1,self.m_item + 1))
                else:
                    self.tokens.append(s)
                
                self.labels.append(s)
            else:
                self.tokens.append(s)
                self.labels.append(0)
        self.tokens = self.tokens[-self.max_len:]
        self.labels = self.labels[-self.max_len:]

        mask_len = self.max_len - len(self.tokens)

        self.tokens = [0] * mask_len + self.tokens
        self.labels = [0] * mask_len + self.labels
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.detach()
        loss = self.calculate_loss()
        loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,loss.cpu().item()

    def calculate_loss(self):
        seqs,labels = torch.tensor(self.tokens).unsqueeze(0).to(args.device),torch.tensor(self.labels).unsqueeze(0).to(args.device)
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss
    
    def eval_(self,model):
        seq = self._train_
        #idx = self.max_len - 1
        answer = self._test_[0]
        #seq[idx] = valid[u][0]
        #idx -= 1
        # for i in reversed(self._train_):
        #     seq[idx] = i
        #     idx -= 1
        #     if idx == -1: break
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._test_[0]]
        for _ in range(1000):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        if self._target_:
            item_idx.append(self._target_[0])
        #print("hahah")
        #predictions = model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
        #predictions = predictions[0] # - for 1st argsort DESC
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        length = len(item_idx)
        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        predictions = model(torch.tensor(seq).unsqueeze(0).to(args.device))[:,-1,:]
        predictions = predictions.gather(1,item_idx)[0]
        if self._test_[0] in rated:
            predictions[0] = - (1 << 10)
        if self._target_[0] in rated:
            predictions[length] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(predictions, [0], 10)
            prec_at_20 = evaluate_precision(predictions, [0], 10)
            ndcg_at_20 = evaluate_ndcg(predictions, [0], 10)
            test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

            #predictions[0] = - (1 << 10)
        else:
            test_result = None

        if self._target_:
            length -= 1
            er_at_5 = evaluate_recall(predictions, [length], 5)
            er_at_10 = evaluate_recall(predictions, [length], 10)
            er_at_20 = evaluate_recall(predictions, [length], 20)
            er_at_30 = evaluate_recall(predictions, [length], 30)
            target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
        else:
            target_result = None

        return test_result, target_result
    
class FedRecSASRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = target_ind
        self.m_item = m_item
        # self.dim = dim
        self.max_len = 200
        self.mask_prob = 0.15
        self.mask_token = m_item + 1
        seed = 2024
        self.rng = random.Random(seed)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.tokens = []
        self.labels = []
        #self.neg = []
        self.labels = self._train_[-self.max_len:]
        self.label_len = len(self.labels)
        self.tokens = self._train_[:-1][-self.max_len:]

        mask_len = self.max_len - len(self.tokens)

        self.tokens = [0] * mask_len + self.tokens
        # while(len(self.neg) < len(self.labels)):
        #     item = self.rng.randint(1,self.m_item + 1)
        #     if item in self._train_ or item in self.neg:
        #         continue
        #     self.neg.append(item)
        mask_len = self.max_len - len(self.labels)
        self.mask_len = mask_len
        self.labels = [0] * mask_len + self.labels
        #self.neg = [0] * mask_len + self.neg 
        self.tokens = torch.tensor(self.tokens).unsqueeze(0).to(args.device)
        self.labels = torch.tensor(self.labels).unsqueeze(0).to(args.device)
        #self.neg = torch.tensor(self.neg).unsqueeze(0).to(args.device)
        self.bce = nn.BCEWithLogitsLoss()
    def train_(self,model):
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.detach()
        self.neg = []
        while(len(self.neg) < self.label_len):
            item = self.rng.randint(1,self.m_item + 1)
            if item in self._train_ or item in self.neg:
                continue
            self.neg.append(item)
        self.neg = [0] * self.mask_len + self.neg 
        self.neg = torch.tensor(self.neg).unsqueeze(0).to(args.device)
        loss = self.calculate_loss()
        loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,loss.cpu().item()
    def train_robust(self,model):
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.detach()
        tokens = self.tokens.clone()
        seq_before = tokens.clone()
        tokens = F.one_hot(tokens,num_classes = self.model.embedding.token.weight.size(0)).float()
        self.model.eval()
        adv_iteration = 3
        adv_step = 1.0
        substitution_ratio = 0.5
        with torch.enable_grad():
            for _ in range(adv_iteration):
                tokens.requires_grad = True
                loss = self.calculate_loss_robust(tokens) # TODO
                loss.backward()
                input_grad = tokens.grad.data / (torch.norm(tokens.grad.data,dim=-1,keepdim=True) + 1e-9)

                tokens = tokens + adv_step * input_grad
                tokens = torch.clamp(tokens,min=0.)
                tokens = tokens / tokens.sum(-1,keepdim=True)
                tokens = tokens.detach()
        switch_indices = (torch.rand(seq_before.shape) <= substitution_ratio).to(args.device)
        switch_indices = (switch_indices * (seq_before != 0)).float().unsqueeze(-1)
        tokens = switch_indices * tokens + (1 - switch_indices) * F.one_hot(seq_before,
        num_classes=self.model.embedding.token.weight.size(0)).float()
        self.model.train()
        #self.optimizer.zero_grad()
        loss = self.calculate_loss_robust(tokens)
        loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,loss.cpu().item()
    def calculate_loss(self):
        #seqs, labels, negs = batch # seq,labels,negs .shape = (64,200)
        # labels为样本正标签 negs为样本负标签
        logits = self.model(self.tokens)  # F.softmax(self.model(seqs), dim=-1) logits.shape = (64,200,3417)
        pos_logits = logits.gather(-1, self.labels.unsqueeze(-1))[self.tokens > 0].squeeze() # 包含根据seqs中非零位置指示的labels 索引从 logits 选择出的值
        pos_targets = torch.ones_like(pos_logits)
        neg_logits = logits.gather(-1, self.neg.unsqueeze(-1))[self.tokens > 0].squeeze()
        neg_targets = torch.zeros_like(neg_logits)

        loss = self.bce(torch.cat((pos_logits, neg_logits), 0), torch.cat((pos_targets, neg_targets), 0))
        return loss
    
    def calculate_loss_robust(self,tokens):
        logits = self.model(tokens)  # F.softmax(self.model(seqs), dim=-1) logits.shape = (64,200,3417)
        #pos_logits = logits.gather(-1, self.labels.unsqueeze(-1))[tokens > 0].squeeze() # 包含根据seqs中非零位置指示的labels 索引从 logits 选择出的值
        #pos_targets = torch.ones_like(pos_logits)
        #neg_logits = logits.gather(-1, self.neg.unsqueeze(-1))[tokens > 0].squeeze()
        #neg_targets = torch.zeros_like(neg_logits)
        pos_logits = logits.gather(-1, self.labels.unsqueeze(-1))[tokens[:, :, 1:].sum(-1) > 0].squeeze()
        neg_logits = logits.gather(-1, self.neg.unsqueeze(-1))[tokens[:, :, 1:].sum(-1) > 0].squeeze()
        pos_targets = torch.ones_like(pos_logits)
        neg_targets = torch.zeros_like(neg_logits)
        loss = self.bce(torch.cat((pos_logits, neg_logits), 0), torch.cat((pos_targets, neg_targets), 0))
        return loss
    def eval_(self,model):
        seq = self._train_
        #idx = self.max_len - 1
        answer = self._test_[0]
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._test_[0]]
        for _ in range(1000):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        if self._target_:
            item_idx.append(self._target_[0])
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        length = len(item_idx)
        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        predictions = model(torch.tensor(seq).unsqueeze(0).to(args.device))[:,-1,:]
        predictions = predictions.gather(1,item_idx)[0]
        if self._test_[0] in rated:
            predictions[0] = - (1 << 10)
        if self._target_[0] in rated:
            predictions[length] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(predictions, [0], 10)
            prec_at_20 = evaluate_precision(predictions, [0], 10)
            ndcg_at_20 = evaluate_ndcg(predictions, [0], 10)
            test_result = np.array([hr_at_20, prec_at_20, ndcg_at_20])

            #predictions[0] = - (1 << 10)
        else:
            test_result = None

        if self._target_:
            length -= 1
            er_at_5 = evaluate_recall(predictions, [length], 5)
            er_at_10 = evaluate_recall(predictions, [length], 10)
            er_at_20 = evaluate_recall(predictions, [length], 20)
            er_at_30 = evaluate_recall(predictions, [length], 30)
            target_result = np.array([er_at_5, er_at_10, er_at_20, er_at_30])
        else:
            target_result = None

        return test_result, target_result
    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)