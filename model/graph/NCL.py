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

class NCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NCL, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config['embbedding_size'],
            self.config['model_config']['num_layers']
        )
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
    # e-step  
    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(
            d=self.config['embbedding_size'],
            k=self.config['model_config']['num_clusters'],
            gpu=True
        )
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = InfoNCE(user_emb[user_idx], user2centroids, self.config['model_config']['temperature']) * self.config['batch_size']
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = InfoNCE(item_emb[item_idx], item2centroids, self.config['model_config']['temperature']) * self.config['batch_size']
        # 注意这里 proto_nce_loss_item 前面没有去加这个 alpha 系数
        proto_nce_loss = self.config['model_config']['proto_reg'] * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    # 结构位置对比损失
    # https://github.com/RUCAIBox/NCL/issues/10
    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.config['model_config']['temperature'])
        ttl_score_user = torch.exp(ttl_score_user / self.config['model_config']['temperature']).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.config['model_config']['temperature'])
        ttl_score_item = torch.exp(ttl_score_item / self.config['model_config']['temperature']).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        pos_loss = self.config['model_config']['ssl_reg'] * (ssl_loss_user + self.config['model_config']['alpha'] * ssl_loss_item)
        return pos_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        for epoch in range(self.config['num_epochs']):
            if epoch >= 20:
                # 大于20轮才开始做e-step
                self.e_step()
            for index, batch in enumerate(next_batch_pairwise(self.data, self.config['batch_size'])):
                user_idx, pos_idx, neg_idx = batch

                # brp 损失 LGCN 部分 emb_list 是 all_emb
                rec_user_emb, rec_item_emb, emb_list = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_bpr_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                # 第一层初始的 GNN emb
                initial_emb = emb_list[0]
                # 第 2*N 层 GNN emb
                context_emb = emb_list[self.config['model_config']['hyper_layers']*2]
                # 这部分是layer位置的对比损失
                pos_loss = self.ssl_layer_loss(context_emb, initial_emb, user_idx,pos_idx)

                # 推荐的brp损失+l2损失
                rec_loss = rec_bpr_loss + l2_reg_loss(self.config['lambda'], user_emb, pos_item_emb, neg_item_emb) / self.config['batch_size']

                # 原型损失
                proto_loss = torch.zeros_like(pos_loss)
                if epoch >= 20: #warm_up
                    proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                    
                cl_loss = proto_loss + pos_loss
                batch_loss = rec_loss + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                wandb.log({
                    'epoch': epoch + 1,
                    'batch_loss': batch_loss.item(),
                    'rec_loss': rec_loss.item(),
                    'cl_loss': cl_loss.item(),
                    'proto_loss': proto_loss.item(),
                    'pos_loss': pos_loss.item()
                })

                if index % 100==0:
                    print('training:', epoch + 1, 'batch', index, 'rec_loss:', rec_loss.item(), 'pos_loss', pos_loss.item(), 'proto_loss', proto_loss.item())
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
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings, all_embeddings
