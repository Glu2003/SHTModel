from helpers.BaseRunner import BaseRunner
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import numpy as np
from models.SHTModel import SHTModel  # 需要先导入 SHTModel
class SHTRunner(BaseRunner):
    def __init__(self, args, data):  # 添加 data 参数
        super().__init__(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model = SHTModel(args, data)  # 在初始化时创建模型

    def evaluate(self, data, epoch, phase="test"):
        """评估模型性能"""
        model = self.model
        model.eval()
        
        # 获取所有用户的预测
        with torch.no_grad():
            for batch_data in data:
                batch_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in batch_data.items()}
                out_dict = model(batch_data)
                user_embeds = out_dict['user_embeds']
                item_embeds = out_dict['item_embeds']
                break  # 只需要一个batch就能获取所有嵌入
                
            all_scores = torch.mm(user_embeds, item_embeds.t())
            
        # 计算评估指标
        metrics = self.calc_metrics(all_scores, data.corpus, phase)
        return metrics
        
    def calc_metrics(self, scores, corpus, phase):
        """计��各种评估指标"""
        metrics = {}
        
        # 获取测试集的真实交互
        if phase == 'test':
            eval_user_set = corpus.test_clicked_set
        elif phase == 'dev' or phase == 'valid':  # 同时支持'dev'和'valid'两种命名
            eval_user_set = corpus.valid_clicked_set  # 从dev_clicked_set改为valid_clicked_set
        else:
            raise ValueError(f"未知的评估阶段: {phase}")
        
        # 计算topk推荐的命中率等指标
        hit_ratio = []
        ndcg = []
        for u in eval_user_set:
            user_scores = scores[u]
            interacted_items = eval_user_set[u]
            
            # 过滤掉训练集中的物品
            if u in corpus.train_clicked_set:
                user_scores[list(corpus.train_clicked_set[u])] = -np.inf
                
            # 获取topk推荐
            _, topk_items = torch.topk(user_scores, k=self.args.topk)
            topk_items = topk_items.cpu().numpy()
            
            # 计算HR@k
            hit_num = len(set(topk_items) & set(interacted_items))
            hit_ratio.append(hit_num / len(interacted_items))
            
            # 计算NDCG@k
            dcg = 0.0
            idcg = sum([1.0/np.log2(i+2) for i in range(min(len(interacted_items), self.args.topk))])
            for i, item in enumerate(topk_items):
                if item in interacted_items:
                    dcg += 1.0/np.log2(i+2)
            ndcg.append(dcg/idcg)
            
        metrics['HR'] = np.mean(hit_ratio)
        metrics['NDCG'] = np.mean(ndcg)
        return metrics 
