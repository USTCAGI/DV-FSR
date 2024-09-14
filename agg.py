import math
import os
import torch
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

def weighted_average_oracle(points, weights):
    """Computes weighted average of models with specified weights.

    Args:
        points: list of dicts, where each dict contains model parameters (layers)
            Each key in the dict is a parameter name, and the value is a tensor.
        weights: list of weights of the same length as points.

    Returns:
        A dict containing the weighted average of the model parameters.
    """
    # Initialize an empty dictionary to store the weighted averages.
    weighted_updates = {}
    #print(points[0].keys())
    for key in points[0].keys():
        #print("hahah",key)
        # Initialize the weighted average tensor as zeros with the same shape and device as the first parameter tensor.
        weighted_updates[key] = torch.zeros_like(points[0][key])
    
    # Calculate the total weight for normalization.
    total_weight = sum(weights)

    # Accumulate weighted updates.
    for point, weight in zip(points, weights):
        for key in point.keys():
            if point[key] is not None:
                weighted_updates[key] += (weight / total_weight) * point[key]

    return weighted_updates

def average(points,weights):
    weighted_updates = {}
    #print(points[0].keys())
    for key in points[0].keys():
        #print("hahah",key)
        # Initialize the weighted average tensor as zeros with the same shape and device as the first parameter tensor.
        weighted_updates[key] = torch.zeros_like(points[0][key])
    
    # Calculate the total weight for normalization.
    total_weight = sum(weights)

    # Accumulate weighted updates.
    for point, weight in zip(points, weights):
        for key in point.keys():
            if point[key] is not None:
                weighted_updates[key] += (weight / total_weight) * point[key]

    return weighted_updates
def l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a dict of tensors representing model parameters"""
    return torch.sqrt(sum((p1[key] - p2[key]).pow(2).sum() for key in p1))

def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective, where each point is a dict of tensors."""
    return sum(alpha * l2dist(median, p) for alpha, p in zip(alphas, points))


class FedAdam():
    def __init__(self,server_model,device):
        self.server_model = server_model
        #LR = 0.001
        #LR = 0.0005
        #self.server_optimizer = optim.Adam(self.server_model.parameters(),lr = LR)
        # self.args = args
        self.device = device
        self._reinit()
    
    def _reinit(self):
        self.client_user_grad = 0
        # self.client_item_grad = []
        # self.client_other_grad = []
        self.client_grad = []
        self.client_sample_num = []
        self.attacker_list = []
        #self.server_optimizer.zero_grad()
        self.batch_model_grad = {}
        for name,param in self.server_model.named_parameters():
            self.batch_model_grad[name] = torch.zeros_like(param.data).to(self.device)

    @torch.no_grad()
    def collect_client_update(self, model_grad_dict,client_sample_num):
        # direct add for user embedding
        # self.client_user_grad += client_user_grad
        # self.client_item_grad.append(client_item_grad)
        # self.client_other_grad.append(client_other_grad)
        # self.client_sample_num.append(client_sample_num)
        # self.attacker_list.append(is_attacker)
        self.client_grad.append(model_grad_dict)
        self.client_sample_num.append(client_sample_num)
        # for name,param in self.batch_model_grad.items():
        #     if(model_grad_dict[name] != None):
        #         self.batch_model_grad[name] += model_grad_dict[name]
    
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_weight = client_sample_num.float() / client_sample_num.sum()

        # client_item_grad = torch.stack(self.client_item_grad, dim=0)
        # agg_client_item_grad = torch.matmul(client_weight, client_item_grad)

        # client_other_grad = torch.stack(self.client_other_grad, dim=0)
        # agg_client_other_grad = torch.matmul(client_weight, client_other_grad)

        # vector_to_grad(self.client_user_grad, self.server_model.user_model)
        # vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        # vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        for name in self.client_grad[0]:  # 假设所有字典都有相同的键
            self.batch_model_grad[name] = torch.zeros_like(self.client_grad[0][name])
        for client_dict, weight in zip(self.client_grad, client_weight):
            for name in client_dict:
                if client_dict[name] != None:
                    self.batch_model_grad[name] += client_dict[name] * weight
        # for name,param in self.server_model.named_parameters():
        #     param.grad = self.batch_model_grad[name]
        # self.server_optimizer.step()
        #self._reinit()

class Geometric_mean():
    def __init__(self,server_model,device):
        self.server_model = server_model
        #LR = 0.001
        #LR = 0.0005
        #self.server_optimizer = optim.Adam(self.server_model.parameters(),lr = LR)
        # self.args = args
        self.device = device
        self._reinit()
    
    def _reinit(self):
        self.client_user_grad = 0
        # self.client_item_grad = []
        # self.client_other_grad = []
        self.client_grad = []
        self.client_sample_num = []
        self.attacker_list = []
        #self.server_optimizer.zero_grad()
        self.batch_model_grad = {}
        self.median = {}
        for name,param in self.server_model.named_parameters():
            self.batch_model_grad[name] = torch.zeros_like(param.data).to(self.device)

    @torch.no_grad()
    def collect_client_update(self, model_grad_dict,client_sample_num):
        # direct add for user embedding
        # self.client_user_grad += client_user_grad
        # self.client_item_grad.append(client_item_grad)
        # self.client_other_grad.append(client_other_grad)
        # self.client_sample_num.append(client_sample_num)
        # self.attacker_list.append(is_attacker)
        self.client_grad.append(model_grad_dict)
        self.client_sample_num.append(client_sample_num)
        # for name,param in self.batch_model_grad.items():
        #     if(model_grad_dict[name] != None):
        #         self.batch_model_grad[name] += model_grad_dict[name]
    
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_weight = client_sample_num.float() / client_sample_num.sum()
        maxiter=4
        eps=1e-5
        verbose=False
        ftol=1e-6
        device = self.client_grad[0][next(iter(self.client_grad[0]))].device
        
        # 将 alphas 转换为张量并归一化
        #alphas = torch.tensor(client_weight, dtype=torch.float32, device=device) / sum(client_weight)
        
        alphas = client_weight / sum(client_weight)
        # 计算初始中位数
        median = weighted_average_oracle(self.client_grad, alphas)
        num_oracle_calls = 1

        # 初始目标函数值
        obj_val = geometric_median_objective(median, self.client_grad, alphas)
        logs = [{'iteration': 0, 'objective_value': obj_val, 'delta': 0, 'distance': 0}]
        
        if verbose:
            print('Starting Weiszfeld algorithm')
            print(logs[0])

        # 开始迭代过程
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            # 更新权重
            weights = torch.tensor([alpha / max(eps, l2dist(median, p).item()) for alpha, p in zip(alphas, self.client_grad)],
                                   dtype=torch.float32, device=device)
            weights /= weights.sum()
            
            # 更新中位数
            median = weighted_average_oracle(self.client_grad, weights)
            num_oracle_calls += 1
            
            # 新的目标函数值
            obj_val = geometric_median_objective(median, self.client_grad, alphas)
            log_entry = {
                'iteration': i+1,
                'objective_value': obj_val,
                'delta': (prev_obj_val - obj_val) / obj_val,
                'distance': l2dist(median, prev_median)
            }
            logs.append(log_entry)
            if verbose:
                print(log_entry)
            
            # 检查收敛条件
            #print(i)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                #print(i,"hahaha")
                break
        self.batch_model_grad = median
        #for name,param in self.server_model.named_parameters():
         #   param.grad = median[name]
        #self.server_optimizer.step()
        # self._reinit()