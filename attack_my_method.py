import torch
import torch.nn as nn
from parse import args
from client import FedRecClient
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict
class SeqAttackClient_method_pipattack(nn.Module):
    def __init__(self, test_items,target_items, m_item, dim,train_id):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._train_ = train_id
        self.max_len = 200
        self.test_item = test_items[0]
    
    # method 1
    def train_(self,model):
        self.model = deepcopy(model)
        #self.item_embeddings = self.model.item_emb.weight
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
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        #s = 10
        #total_loss = 0

        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        
        #predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
        #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
        predictions = self.model(torch.tensor(seq).unsqueeze(0).to(args.device))[:,-1,:]
        predictions = predictions.gather(1,item_idx)[0]
        #loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
        loss = -torch.log(predictions.sigmoid()[0]) # pip attack
        #print(loss)
        #total_loss += (1 / s) * loss
        #total_loss.backward()
        loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None
    
    def eval_(self,model):
        return None, None
class SeqAttackClient_method1(nn.Module):
    def __init__(self, test_items,target_items, m_item, dim,train_id):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._train_ = train_id
        self.max_len = 200
        self.test_item = test_items[0]
    
    # method 1
    def train_(self,model):
        self.model = deepcopy(model)
        #self.item_embeddings = self.model.item_emb.weight
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
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        s = 10
        total_loss = 0

        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        for _ in range(s):
            #predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            predictions = self.model(torch.tensor(seq).unsqueeze(0).to(args.device))[:,-1,:]
            predictions = predictions.gather(1,item_idx)[0]
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            #print(loss)
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None
    def eval_(self,model):
        return None, None

class SeqAttackClient_method2(nn.Module):
    def __init__(self, test_items,target_items, m_item, dim,train_id):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._train_ = train_id
        self.max_len = 200
        self.test_item = test_items[0]
    def contrastive_loss(self,user_repr, pos_item, neg_items):
        """
        Calculate contrastive loss.
    
        Args:
        - user_repr: Tensor representing the user embeddings [batch_size, embedding_dim]
        - pos_item: Tensor representing positive item embeddings [batch_size, embedding_dim]
        - neg_items: Tensor representing negative item embeddings [batch_size, num_negatives, embedding_dim]
    
        Returns:
        - loss: Calculated contrastive loss
        """
        # Calculate the score for the positive item
        pos_scores = torch.sum(user_repr * pos_item, dim=1)  # Dot product
    
        # Calculate the scores for the negative items
        neg_scores = torch.bmm(neg_items, user_repr.unsqueeze(2)).squeeze(2)  # Batch matrix-matrix product
    
        # Concatenate scores
        scores = torch.cat((pos_scores.unsqueeze(1), neg_scores), dim=1)
    
        # Compute softmax over scores
        logits = F.log_softmax(scores, dim=1)
    
        # Negative log-likelihood loss for the first class (positive class)
        loss = -logits[:, 0].mean()  # Taking the negative log probability of the positive class
    
        return loss
    def contrastive_2(self, anchors, positives, negatives):
        # 计算锚点与正样本之间的余弦相似度
        pos_similarity = F.cosine_similarity(anchors, positives)
        # 计算锚点与每个负样本之间的余弦相似度
        neg_similarity = F.cosine_similarity(anchors.unsqueeze(1), negatives, dim=2)
        # 应用对数 softmax
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss.mean()

    # method2
    def get_frequent_items(self,train_id):
    # 初始化字典来存储物品的交互次数
        item_count = defaultdict(int)

        # 计算每个物品的交互次数
        for item_id in train_id:
            item_count[item_id] += 1

     # 找到交互次数最多的物品
        max_count = max(item_count.values())
        most_frequent_items = [item for item, count in item_count.items() if count == max_count]

        return most_frequent_items
    def train_(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.embedding.token
       
        
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
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        neg_item_array = item_idx[1:]
        #least_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(neg_item_array).to(args.device)),dim=0).unsqueeze(0)
        most_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(self._train_).to(args.device)),dim=0).unsqueeze(0)
        #most_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(self.get_frequent_items(self._train_)).to(args.device)),dim=0).unsqueeze(0)
        #most_liked_embeddings = self.item_embeddings(torch.tensor(self._train_[-1]).to(args.device)).unsqueeze(0)
        #most_liked_embeddings = self.item_embeddings(torch.tensor(self.test_item).to(args.device)).unsqueeze(0)
        least_liked_embeddings = self.item_embeddings(torch.tensor(neg_item_array).to(args.device)).unsqueeze(0)
        s = 10
        total_loss = 0
        target_embedding = self.item_embeddings(torch.tensor(self._target_[0]).to(args.device)).unsqueeze(0)
        for _ in range(s):
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            loss = 0.0
            #positive_loss = criterion(target_embedding, most_liked_embeddings, torch.tensor([1]).to(args.device))
            #negative_loss = criterion(target_embedding, least_liked_embeddings, torch.tensor([-1]).to(args.device))
            #loss += self.contrastive_loss(target_embedding,most_liked_embeddings,least_liked_embeddings)
            loss += self.contrastive_2(target_embedding,most_liked_embeddings,least_liked_embeddings) # 超参设置
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            # if param.grad == None:
            #     print(name)
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None
    def eval_(self,model):
        return None, None


class SeqAttackClient_method2_1(nn.Module):
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
    def contrastive_2(self, anchors, positives, negatives):
        # 计算锚点与正样本之间的余弦相似度
        pos_similarity = F.cosine_similarity(anchors, positives)
        # 计算锚点与每个负样本之间的余弦相似度
        neg_similarity = F.cosine_similarity(anchors.unsqueeze(1), negatives, dim=2)
        # 应用对数 softmax
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        labels[0] = 1
        loss = F.cross_entropy(logits, labels)
        return loss.mean()
    def train_(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.embedding.token
        
        for param in self.model.parameters():
            param.detach()
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        neg_item_array = item_idx[1:]
        #least_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(neg_item_array).to(args.device)),dim=0).unsqueeze(0)
        most_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(self._train_).to(args.device)),dim=0).unsqueeze(0)
        least_liked_embeddings = self.item_embeddings(torch.tensor(neg_item_array).to(args.device)).unsqueeze(0)
        s = 10
        total_loss = 0
        target_embedding = self.item_embeddings(torch.tensor(self._target_[0]).to(args.device)).unsqueeze(0)
        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        for _ in range(s):
            #predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            #loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            predictions = self.model(torch.tensor(seq).unsqueeze(0).to(args.device))[:,-1,:]
            predictions = predictions.gather(1,item_idx)[0]
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            #positive_loss = criterion(target_embedding, most_liked_embeddings, torch.tensor([1]).to(args.device))
            #negative_loss = criterion(target_embedding, least_liked_embeddings, torch.tensor([-1]).to(args.device))
            #loss += positive_loss + negative_loss
            loss += self.contrastive_2(target_embedding,most_liked_embeddings,least_liked_embeddings)
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None
    def eval_(self,model):
        return None, None
class SeqAttackClient_method1_3(nn.Module):
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

    def train_(self,model):
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.detach()
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        # profile pollution attack
        seq = torch.tensor(seq,dtype=torch.int64).unsqueeze(0).to(args.device)
        
        #借助这个去做model poisoning
        # model posioning attack
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        s = 10
        total_loss = 0
        is_subsitition = True
        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        for _ in range(s):
            if is_subsitition == True:
                #seq = self.substition_attack(seq)
                perturbed_seq = self.substition_attack(seq)
            # perturbed_seq[0,-1] = self._target_[0]
            #predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            #loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            predictions = self.model(perturbed_seq.to(args.device))[:,-1,:]
            #predictions = self.model(seq.to(args.device))[:,-1,:]
            predictions = predictions.gather(1,item_idx)[0]
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None

    def substition_attack(self,seq):
        self.item_embeddings = self.model.embedding.token.weight.detach().cpu().numpy()[1:]
        self.item_embeddings = torch.tensor(self.item_embeddings).to(args.device) #(3708,64)
        min_cos_sim=0.5 
        repeated_search=10
        self.model.eval()
        with torch.no_grad():
            perturbed_seqs = seq.clone()
            #embeddings,timeline_mask= self.model.SASEmbedding(perturbed_seqs.long())
            embeddings,mask = self.model.embedding(perturbed_seqs.long())
        self.model.train()
        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        scores = self.model.model(embeddings,self.model.embedding.token.weight,mask)[:,-1,:]
        loss = self.adv_ce(scores, torch.tensor([self._target_[0]] * perturbed_seqs.size(0)).to(args.device))
        self.model.zero_grad()
        loss.backward()
        embeddings_grad = embeddings.grad.data
        importance_scores = torch.norm(embeddings_grad, dim=-1)
        self.model.eval()
        #num_attack = 2
        num_attack = args.num_attack
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
                all_embeddings = self.item_embeddings.unsqueeze(1).repeat_interleave(current_embeddings.size(0),1)
                cos = nn.CosineSimilarity(dim = -1,eps=1e-6)
                multipication_results = torch.t(cos(current_embeddings - current_embeddings_grad.sign(),all_embeddings))
                cos_filter_results = cos(all_embeddings,current_embeddings)
                cos_filter_results = torch.t(cos_filter_results >= min_cos_sim)
                multipication_results = multipication_results * cos_filter_results
                _,candidate_indices = torch.sort(multipication_results,dim=1,descending=True)

                if_prev_target = (best_seqs[row_indices,col_indices - 1] == self._target_[0])
                multipication_results[:,self._target_[0] - 1] = multipication_results[:,self._target_[0] - 1] + (if_prev_target * 1e-9)
                _,candidate_indices = torch.sort(multipication_results,dim=1,descending=True)

                best_seqs[row_indices,col_indices] = candidate_indices[:,0] + 1
                logits = F.softmax(self.model(best_seqs)[:,-1,:],dim=-1)
                best_scores = torch.gather(logits,-1,torch.tensor([self._target_[0]] * best_seqs.size(0)).unsqueeze(1).to(args.device)).squeeze()
                for time in range(1,repeated_search):
                    temp_seqs = best_seqs.clone().detach()
                    temp_seqs[row_indices,col_indices] = candidate_indices[:,time] + 1
                    logits = F.softmax(self.model(temp_seqs)[:,-1,:],dim=-1)
                    temp_scores = torch.gather(logits, -1, torch.tensor([self._target_[0]] * best_seqs.size(0)).unsqueeze(1).to(args.device)).squeeze()
                    best_seqs[row_indices, col_indices] = temp_seqs[row_indices, col_indices] * (temp_scores >= best_scores) + best_seqs[row_indices, col_indices] * (temp_scores < best_scores)
                    best_scores = temp_scores * (temp_scores >= best_scores) + best_scores * (temp_scores < best_scores)
                    best_seqs = best_seqs.detach()
                    best_scores = best_scores.detach()
                    del temp_scores
        perturbed_seqs = best_seqs.detach()
        return perturbed_seqs
    def eval_(self,model):
        return None, None
    

class SeqAttackClient_method1_2_3(nn.Module):
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

    def contrastive_2(self, anchors, positives, negatives):
        # 计算锚点与正样本之间的余弦相似度
        pos_similarity = F.cosine_similarity(anchors, positives)
        # 计算锚点与每个负样本之间的余弦相似度
        neg_similarity = F.cosine_similarity(anchors.unsqueeze(1), negatives, dim=2)
        # 应用对数 softmax
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss.mean()
    
    def train_(self,model):
        self.model = deepcopy(model)
        self.item_embeddings = self.model.embedding.token
        for param in self.model.parameters():
            param.detach()
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        #seq[idx] = valid[u][0]
        #idx -= 1
        for i in reversed(self._train_):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        # profile pollution attack
        seq = torch.tensor(seq,dtype=torch.int64).unsqueeze(0).to(args.device)
        
        #借助这个去做model poisoning
        # model posioning attack
        rated = set(self._train_)
        #item_idx = [test[u][0]]
        item_idx = [self._target_[0]]
        for _ in range(100):
            t = np.random.randint(1, self.m_item + 1)
            while t in rated: t = np.random.randint(1, self.m_item + 1)
            item_idx.append(t)
        neg_item_array = item_idx[1:]
        #least_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(neg_item_array).to(args.device)),dim=0).unsqueeze(0)
        most_liked_embeddings = torch.mean(self.item_embeddings(torch.tensor(self._train_).to(args.device)),dim=0).unsqueeze(0)
        least_liked_embeddings = self.item_embeddings(torch.tensor(neg_item_array).to(args.device)).unsqueeze(0)
        s = 10
        total_loss = 0
        target_embedding = self.item_embeddings(torch.tensor(self._target_[0]).to(args.device)).unsqueeze(0)
        is_subsitition = True
        item_idx = torch.tensor(item_idx).unsqueeze(0).to(args.device)
        for _ in range(s):
            if is_subsitition == True:
                #seq = self.substition_attack(seq)
                perturbed_seq = self.substition_attack(seq)
            #predictions = self.model.predict(*[np.array(l) for l in [[1],[seq], item_idx]])[0]
            #print(type(predictions.sigmoid()[0]),type(self._target_),predictions.sigmoid()[0].shape,predictions.sigmoid()[0].unsqueeze(0).shape)
            #loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            predictions = self.model(perturbed_seq.to(args.device))[:,-1,:]
            #predictions = self.model(seq.to(args.device))[:,-1,:]
            predictions = predictions.gather(1,item_idx)[0]
            loss = nn.BCELoss()(predictions.sigmoid()[0].unsqueeze(0), torch.ones(len(self._target_)).to(args.device))
            loss += self.contrastive_2(target_embedding,most_liked_embeddings,least_liked_embeddings)
            total_loss += (1 / s) * loss
        total_loss.backward()
        model_grad_dict = {}
        model_grad_array = []
        for name,param in self.model.named_parameters():
            model_grad_dict[name] = param.grad
            model_grad_array.append(param.grad)
        return self._train_,model_grad_array,model_grad_dict,None

    def substition_attack(self,seq):
        self.item_embeddings_subs = self.model.embedding.token.weight.detach().cpu().numpy()[1:]
        self.item_embeddings_subs = torch.tensor(self.item_embeddings_subs).to(args.device)
        min_cos_sim=0.5 
        repeated_search=10
        self.model.eval()
        with torch.no_grad():
            perturbed_seqs = seq.clone()
            #embeddings,timeline_mask= self.model.SASEmbedding(perturbed_seqs.long())
            embeddings,mask = self.model.embedding(perturbed_seqs.long())
        self.model.train()
        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        scores = self.model.model(embeddings,self.model.embedding.token.weight,mask)[:,-1,:]
        loss = self.adv_ce(scores, torch.tensor([self._target_[0]] * perturbed_seqs.size(0)).to(args.device))
        self.model.zero_grad()
        loss.backward()
        embeddings_grad = embeddings.grad.data
        importance_scores = torch.norm(embeddings_grad, dim=-1)
        self.model.eval()
        #num_attack = 2
        num_attack = args.num_attack
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
                all_embeddings = self.item_embeddings_subs.unsqueeze(1).repeat_interleave(current_embeddings.size(0),1)
                cos = nn.CosineSimilarity(dim = -1,eps=1e-6)
                multipication_results = torch.t(cos(current_embeddings - current_embeddings_grad.sign(),all_embeddings))
                cos_filter_results = cos(all_embeddings,current_embeddings)
                cos_filter_results = torch.t(cos_filter_results >= min_cos_sim)
                multipication_results = multipication_results * cos_filter_results
                _,candidate_indices = torch.sort(multipication_results,dim=1,descending=True)

                if_prev_target = (best_seqs[row_indices,col_indices - 1] == self._target_[0])
                multipication_results[:,self._target_[0] - 1] = multipication_results[:,self._target_[0] - 1] + (if_prev_target * 1e-9)
                _,candidate_indices = torch.sort(multipication_results,dim=1,descending=True)

                best_seqs[row_indices,col_indices] = candidate_indices[:,0] + 1
                logits = F.softmax(self.model(best_seqs)[:,-1,:],dim=-1)
                best_scores = torch.gather(logits,-1,torch.tensor([self._target_[0]] * best_seqs.size(0)).unsqueeze(1).to(args.device)).squeeze()
                for time in range(1,repeated_search):
                    temp_seqs = best_seqs.clone().detach()
                    temp_seqs[row_indices,col_indices] = candidate_indices[:,time] + 1
                    logits = F.softmax(self.model(temp_seqs)[:,-1,:],dim=-1)
                    temp_scores = torch.gather(logits, -1, torch.tensor([self._target_[0]] * best_seqs.size(0)).unsqueeze(1).to(args.device)).squeeze()
                    best_seqs[row_indices, col_indices] = temp_seqs[row_indices, col_indices] * (temp_scores >= best_scores) + best_seqs[row_indices, col_indices] * (temp_scores < best_scores)
                    best_scores = temp_scores * (temp_scores >= best_scores) + best_scores * (temp_scores < best_scores)
                    best_seqs = best_seqs.detach()
                    best_scores = best_scores.detach()
                    del temp_scores
        perturbed_seqs = best_seqs.detach()
        return perturbed_seqs
    def eval_(self,model):
        return None, None
    
