import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor

# Paper: self-supervised graph learning for recommendation. SIGIR'21


class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        # config data 
        super(SGL, self).__init__(conf, training_set, test_set)
        self.model = SGL_Encoder(
            self.data,
            self.config['embbedding_size'],
            self.config['model_config']['droprate'],
            self.config['model_config']['num_layers'],
            self.config['model_config']['temperature'],
            self.config['model_config']['augtype'],
        )

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        for epoch in range(self.config['num_epochs']):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for index, batch in enumerate(next_batch_pairwise(self.data, self.config['batch_size'])):
                user_idx, pos_idx, neg_idx = batch # neg_idx 是在当前 user 没有交互过的item
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # 有三部分 Loss。推荐系统BPR损失 对比学习损失 L2正则损失
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss = self.config['learning_rate'] * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                # FIXME: 这里是否缺少了参数 lambda???
                l2_loss = l2_reg_loss(self.config['lambda'], user_emb, pos_item_emb, neg_item_emb) 
                batch_loss = rec_loss + l2_loss + cl_loss

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                wandb.log({'epoch':epoch+1,'batch_loss': batch_loss.item(),'rec_loss':rec_loss.item(),'cl_loss': cl_loss.item()})
                if index % 100 == 0:
                    print('training:', epoch + 1, 'batch', index, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch>=5:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    # 在 graph_recommender里面用到(fast_evaluation函数)
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
    # 在 graph_recommender里面用到(test函数)
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        # normalize_graph_mat(self.ui_adj)
        self.norm_adj = data.norm_adj # norm_adj 是图的正则化矩阵D^(-1/2)*A*D^(-1/2)，
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_ # 使用Glorot initialization初始化参数
        embedding_dict = nn.ParameterDict({
            # 大小是 user_num * emb_size, 其中emb_size是D，即向量的长度64
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    # 前向 model() 调用的结果
    # 一般来说对于模型内部使用 self.forward 比较好，外部使用 model()
    def forward(self, perturbed_adj=None):
        # WARNING: 这个部分请参考 LightGCN 的实现，是完全对应上的！
        # self.embedding_dict['user_emb'] 是n*D的
        # self.embedding_dict['item_emb'] 是m*D的
        # 大小是 (n+m)*D 这里相当于是没有带A的纯特征矩阵R。
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings] # 包含k层GNN的embedding
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    # perturbed_adj 已经做过normalize正则化的A矩阵
                    # 这里是 SGL 的实现，实际要在
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                # 这里sparse_norm_adj是普通的正则化的A矩阵 这里是 lightGCN 的操作简化了操作
                # 看他的矩阵形式部分实际上就是  E^(k+1) = D^(-1/2)·A·D^(-1/2)*E^(k)
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        # 拼接向量 all_embeddings size k 个 (n,n) 其中k是卷积网络的层数，n是用户数+物品数
        # 这里请看论文 这里是取了 1/1+L
        all_embeddings = torch.stack(all_embeddings, dim=1) # https://zhuanlan.zhihu.com/p/354177500
        all_embeddings = torch.mean(all_embeddings, dim=1) # n*k*n
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1, view2, self.temp) # 这里是图级别的对比
