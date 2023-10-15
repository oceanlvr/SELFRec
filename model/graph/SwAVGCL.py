import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender

from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import faiss
import wandb

# paper: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW'22
# 为了识别节点 (用户和项目)的语义邻居，HCCF 和 SwAVGCL 追求图结构相邻节点和语义邻居之间的一致表示。

class SwAVGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SwAVGCL, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config['embedding_size'],
            self.config['model_config.num_layers']
        )

    def SwaVNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(
            initial_emb, [self.data.user_num, self.data.item_num])
        
        # your code here
        temperature = 0.1  # you might want to tune this hyperparameter
        
        # Calculate similarities for users
        user_similarity = torch.mm(user_emb, user_emb.t())
        user_similarity /= temperature
        
        # Calculate the swav loss for users
        user_loss = -F.log_softmax(user_similarity, dim=1)[range(len(user_emb)), user_idx]
        swav_nce_loss_user = user_loss.mean()
        
        # Calculate similarities for items
        item_similarity = torch.mm(item_emb, item_emb.t())
        item_similarity /= temperature
        
        # Calculate the swav loss for items
        item_loss = -F.log_softmax(item_similarity, dim=1)[range(len(item_emb)), item_idx]
        swav_nce_loss_item = item_loss.mean()
        
        swav_nce_loss = self.config['model_config.swav_reg'] * (swav_nce_loss_user + swav_nce_loss_item)
        
        return swav_nce_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config['learning_rate'])
        for epoch in range(self.config['num_epochs']):

            for index, batch in enumerate(next_batch_pairwise(self.data, self.config['batch_size'])):
                user_idx, pos_idx, neg_idx = batch

                # brp 损失 LGCN 部分 emb_list 是 all_emb
                rec_user_emb, rec_item_emb, emb_list = model()

                # 第一层初始的 GNN emb
                initial_emb = emb_list[0]
                # 第 2*N 层 GNN emb
                context_emb = emb_list[self.config['model_config.hyper_layers']*2]
                # 这部分是layer位置的对比损失

                # 推荐的brp损失+l2损失
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[
                    user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(
                    self.config['lambda'], user_emb, pos_item_emb, neg_item_emb) / self.config['batch_size']

                cl_loss = self.SwaVNCE_loss(
                    initial_emb, user_idx, pos_idx
                )

                batch_loss = rec_loss + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                wandb.log({
                    'epoch': epoch + 1,
                    'batch_loss': batch_loss.item(),
                    'rec_loss': rec_loss.item(),
                    'cl_loss': cl_loss.item(),
                })

                if index % 100 == 0:
                    print(
                        'training:', epoch + 1, 'batch', index, 'rec_loss:',
                        rec_loss.item(), 'cl_loss', cl_loss.item()
                    )
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(
            self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(
                self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings, all_embeddings
