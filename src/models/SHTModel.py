from models.BaseModel import GeneralModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math

class SHTModel(GeneralModel):
    reader, runner = 'BaseReader', 'BaseRunner'
    
    @staticmethod
    def parse_model_args(parser):
        
        parser.add_argument('--batch', default=4096, type=int, help='batch size')
        parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
        parser.add_argument('--tstBat', default=64, type=int, help='number of users in a testing batch')
        parser.add_argument('--reg', default=1e-8, type=float, help='weight decay regularizer')
        parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
        parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
        parser.add_argument('--latdim', default=64, type=int, help='embedding size')
        parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
        parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
        parser.add_argument('--load_model', default=None, help='model name to load')
        parser.add_argument('--att_head', default=4, type=int, help='number of attention heads')
        parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
        parser.add_argument('--temp', default=1, type=float, help='temperature')
        parser.add_argument('--gcn_hops', default=2, type=int, help='number of hops in gcn precessing')
        parser.add_argument('--mult', default=1, type=float, help='multiplication factor')
        parser.add_argument('--ssl1_reg', default=1, type=float, help='weight for ssl loss')
        parser.add_argument('--ssl2_reg', default=1, type=float, help='weight for ssl loss')
        parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
        parser.add_argument('--edgeSampRate', default=0.1, type=float, help='Ratio of sampled edges')
        return GeneralModel.parse_model_args(parser)
        
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.reg = args.reg
        self.latdim = args.latdim
        self.hyperNum = args.hyperNum
        self.gcn_hops = args.gcn_hops
        self.temp = args.temp
        self.ssl1_reg = args.ssl1_reg
        self.ssl2_reg = args.ssl2_reg
        self.edgeSampRate = args.edgeSampRate
        self.att_head = args.att_head
        self._define_params()
        
    def _define_params(self):
        init = nn.init.xavier_uniform_
        self.uEmbeds = nn.Parameter(init(torch.empty(self.user_num, self.latdim))) 
        self.iEmbeds = nn.Parameter(init(torch.empty(self.item_num, self.latdim)))
        self.uHyper = nn.Parameter(init(torch.empty(self.hyperNum, self.latdim)))
        self.iHyper = nn.Parameter(init(torch.empty(self.hyperNum, self.latdim)))
        
        # 添加多头注意力所需的参数
        self.n_heads = self.att_head  # 使用已有的att_head参数
        head_dim = self.latdim // self.n_heads
        self.W_Q = nn.Parameter(init(torch.empty(self.latdim, self.n_heads * head_dim)))
        self.W_K = nn.Parameter(init(torch.empty(self.latdim, self.n_heads * head_dim)))
        self.W_V = nn.Parameter(init(torch.empty(self.latdim, self.n_heads * head_dim)))
        self.W_O = nn.Parameter(init(torch.empty(self.n_heads * head_dim, self.latdim)))

    def gcnLayer(self, adj, embeds):
        """
        实现GCN层的传播
        adj: 归一化的邻接矩阵
        embeds: 节点嵌入
        """
        return torch.spmm(adj, embeds)
        
    def hgnnLayer(self, embeds, hyper):
        """
        实现多头注意力的超图层
        embeds: 节点嵌入 [num_nodes, latdim]
        hyper: 超边嵌入 [hyperNum, latdim]
        """
        batch_size = embeds.size(0)
        head_dim = self.latdim // self.n_heads

        # 计算Q, K, V
        Q = embeds @ self.W_Q  # [num_nodes, n_heads * head_dim]
        K = hyper @ self.W_K   # [hyperNum, n_heads * head_dim]
        V = hyper @ self.W_V   # [hyperNum, n_heads * head_dim]

        # 重塑张量以支持多头
        Q = Q.view(batch_size, self.n_heads, head_dim)
        K = K.view(self.hyperNum, self.n_heads, head_dim)
        V = V.view(self.hyperNum, self.n_heads, head_dim)

        # 计算注意力分数
        scores = torch.einsum('bhd,nhd->bnh', Q, K)  # [batch_size, n_heads, hyperNum]
        scores = scores / math.sqrt(head_dim)  # 缩放因子
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, hyperNum]

        # 加权聚合
        out = torch.einsum('bnh,nhd->bhd', attention_weights, V)  # [batch_size, n_heads, head_dim]
        
        # 合并多头的结果
        out = out.reshape(batch_size, self.n_heads * head_dim)  # [batch_size, n_heads * head_dim]
        
        # 最终的线性变换
        out = out @ self.W_O  # [batch_size, latdim]

        return out
        '''return embeds @ (hyper.T @ hyper)'''
    def forward(self, feed_dict):
        adj = feed_dict['adj']  # 归一化的邻接矩阵
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], dim=0)  # 初始嵌入
        
        # GCN层的实现: Ē = Ā·Ā^T·E(u) + Ā·E(v)
        lats = [embeds]
        for _ in range(self.gcn_hops):
            # 第一次传播: Ā·E(v)
            first_prop = self.gcnLayer(adj, lats[-1])
            # 第二次传播: Ā·Ā^T·E(u)
            second_prop = self.gcnLayer(adj.t(), first_prop)
            # 组合两次传播的结果
            new_embeds = first_prop + second_prop
            lats.append(new_embeds)
     
        # 残差连接：合并所有层的输出
        embeds = sum(lats)

        # 克隆基础嵌入用于后续处理
        with torch.no_grad():
            base_embeds = embeds.clone()
    
        # 超图注意力处理
        hyperUEmbeds = self.hgnnLayer(base_embeds[:self.user_num], self.uHyper)
        hyperIEmbeds = self.hgnnLayer(base_embeds[self.user_num:], self.iHyper)
        
        # 处理批次数据
        user_ids = feed_dict['user_id'].to(self.device)
        item_ids = feed_dict['item_id'].to(self.device)
        
        # 确保维度正确
        if user_ids.dim() == 0:
            user_ids = user_ids.unsqueeze(0)
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(0)
        
        batch_size = user_ids.size(0)
        n_items = item_ids.size(1)
        
        # 获取批次的嵌入
        batch_user_embeds = hyperUEmbeds[user_ids]
        flat_item_ids = item_ids.reshape(-1)
        flat_item_embeds = hyperIEmbeds[flat_item_ids]
        batch_item_embeds = flat_item_embeds.reshape(batch_size, n_items, -1)
        
        # 计算预测分数
        prediction = (batch_user_embeds.unsqueeze(1) * batch_item_embeds).sum(-1)
        
        return {
            'prediction': prediction,
            'user_embeds': hyperUEmbeds,
            'item_embeds': hyperIEmbeds,
            'embeds': embeds,
        }
    
    def calc_reg_loss(self):
        """计算L2正则化损失"""
        ret = 0
        for W in self.parameters():
            ret += W.norm(2).square()
        return ret
    def loss(self, out_dict):
        """计算总损失，包括BPR损失、SSL损失和正则化损失"""
        prediction = out_dict['prediction']
        user_embeds = out_dict['user_embeds']    # hyperUEmbeds
        item_embeds = out_dict['item_embeds']    # hyperIEmbeds
        embeds = out_dict['embeds']              # 基础嵌入
        
        # 分离基础嵌入为用户和物品嵌入
        base_u_embeds = embeds[:self.user_num]
        base_i_embeds = embeds[self.user_num:]

        # 计算BPR损失
        pos_pred = prediction[:, 0]
        neg_pred = prediction[:, 1:]
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred - neg_pred) + 1e-8))

        # 计算SSL损失
        ssl_loss = 0
        if self.ssl1_reg != 0 or self.ssl2_reg != 0:
            # 随机采样边
            sample_size = int(self.edgeSampRate * self.user_num * 2)
            if sample_size % 2 == 1:
                sample_size += 1
            
            # 随机选择用户和物品
            pck_users = torch.randint(self.user_num, [sample_size], device=self.device)
            pck_items = torch.randint(self.item_num, [sample_size], device=self.device)
            
            # 计算两种嵌入空间的分数
            scores1 = (user_embeds[pck_users] * item_embeds[pck_items]).sum(-1)
            scores2 = (base_u_embeds[pck_users] * base_i_embeds[pck_items]).sum(-1)
            
            # 分割分数用于对比
            half_num = scores1.shape[0] // 2
            fst_scores1, scd_scores1 = scores1[:half_num], scores1[half_num:]
            fst_scores2, scd_scores2 = scores2[:half_num], scores2[half_num:]
            
            # 使用温度参数计算sigmoid分数
            scores1 = ((fst_scores1 - scd_scores1) / self.temp).sigmoid()
            scores2 = ((fst_scores2 - scd_scores2) / self.temp).sigmoid()
            
            # 计算两个方向的SSL损失
            ssl_loss1 = -(scores2.detach() * (scores1 + 1e-8).log() + 
                        (1 - scores2.detach()) * (1 - scores1 + 1e-8).log()).mean() * self.ssl1_reg
            ssl_loss2 = -(scores1.detach() * (scores2 + 1e-8).log() + 
                        (1 - scores1.detach()) * (1 - scores2 + 1e-8).log()).mean() * self.ssl2_reg
            
            ssl_loss = ssl_loss1 + ssl_loss2

        # 计算正则化损失
        reg_loss = self.calc_reg_loss() * self.reg
        
        # 合并所有损失
        total_loss = bpr_loss + ssl_loss + reg_loss
        
        return total_loss
    # 添加预测方法
    def predict(self, feed_dict):
        """模型预测"""
        out_dict = self.forward(feed_dict)
        return out_dict['user_embeds'], out_dict['item_embeds']
    
    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase: str):
            super().__init__(model, corpus, phase)
            self.adj = self._build_adj_matrix()
            
        def _build_adj_matrix(self):
            """构建用户-物品二分图的邻接矩阵"""
            n_users = self.model.user_num
            n_items = self.model.item_num
            
            # 使用列表推导式优化数据收集
            interactions = [(u, i) for u in range(n_users) 
                           if u in self.corpus.train_clicked_set 
                           for i in self.corpus.train_clicked_set[u]]
            
            if not interactions:
                user_ids, item_ids = [], []
            else:
                user_ids, item_ids = zip(*interactions)
            
            # 创建用户-物品交互矩阵
            ui_mat = sp.coo_matrix(
                (np.ones_like(user_ids, dtype=np.float32), 
                (user_ids, item_ids)), 
                shape=(n_users, n_items)
            ).tocsr()  # 转换为CSR格式以支持矩阵操作
            
            # 创建零矩阵块
            user_user = sp.csr_matrix((n_users, n_users))
            item_item = sp.csr_matrix((n_items, n_items))
            
             # 构建完整的邻接矩阵
            top_half = sp.hstack([user_user, ui_mat])
            bottom_half = sp.hstack([ui_mat.transpose(), item_item])
            mat = sp.vstack([top_half, bottom_half])
            
            # 归一化处理
            degree = np.array(mat.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(degree + 1e-8, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat = sp.diags(d_inv_sqrt)
            
            # 计算归一化邻接矩阵
            norm_adj = d_mat @ mat @ d_mat
            
            # 转换为PyTorch稀疏张量
            norm_adj = norm_adj.tocoo()
            indices = np.vstack((norm_adj.row, norm_adj.col))
            indices = torch.from_numpy(indices).long()
            values = torch.from_numpy(norm_adj.data).float()
            
            return torch.sparse_coo_tensor(
                indices, values, 
                torch.Size(norm_adj.shape)
            ).to(self.model.device)
                
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['adj'] = self.adj
            return feed_dict