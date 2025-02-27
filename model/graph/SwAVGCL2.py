import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender

from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import wandb


class SwAVGCL2(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SwAVGCL2, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config["embedding_size"],
            self.config["model_config.num_layers"],
            self.config["model_config.num_clusters"],
        )

    def cal_cl_loss(self, idx, user_emb, item_emb, prototypes, temperature):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        rec_user_emb, cl_user_emb = user_emb[0], user_emb[1]
        rec_item_emb,cl_item_emb = item_emb[0], item_emb[1]
        user_prototypes, item_prototypes = prototypes[0], prototypes[1]

        user_z = torch.cat([rec_user_emb[u_idx], cl_user_emb[u_idx]], dim=0)
        item_z = torch.cat([rec_item_emb[i_idx], cl_item_emb[i_idx]], dim=0)

        user_loss = self.swav_loss(user_z, user_prototypes, temperature=temperature)
        item_loss = self.swav_loss(item_z, item_prototypes, temperature=temperature)
        cl_loss = user_loss + item_loss
        return self.config['model_config.proto_reg'] * cl_loss

    def swav_loss(self, z, prototypes, temperature=0.1):
        # Compute scores between embeddings and prototypes
        # 3862x64 and 2000x64
        scores = torch.mm(z, prototypes.T)

        score_t = scores[: z.size(0) // 2]
        score_s = scores[z.size(0) // 2 :]

        # Apply the Sinkhorn-Knopp algorithm to get soft cluster assignments
        q_t = self.sinkhorn_knopp(score_t)
        q_s = self.sinkhorn_knopp(score_s)

        log_p_t = torch.log_softmax(score_t / temperature + 1e-7, dim=1)
        log_p_s = torch.log_softmax(score_s / temperature + 1e-7, dim=1)
        # print('score_t.shape',score_t.shape)
        # print('log_p_t.shape',log_p_t.shape)
        # print('q_s.shape',q_s.shape)
        # score_t.shape torch.Size([1909, 2000])
        # log_p_t.shape torch.Size([1909, 2000])
        # q_s.shape torch.Size([1909, 2000])

        # Calculate cross-entropy loss
        loss_t = torch.mean(
            -torch.sum(
                q_s * log_p_t,
                dim=1,
            )
        )
        loss_s = torch.mean(
            -torch.sum(
                q_t * log_p_s,
                dim=1,
            )
        )
        # SwAV loss is the average of loss_t and loss_s
        swav_loss = (loss_t + loss_s) / 2
        return swav_loss

    def sinkhorn_knopp(self, scores, epsilon=0.05, n_iters=3):
        with torch.no_grad():
            # 我们通常希望返回的矩阵Q的每一列代表一个样本的软聚类分配，而每一行对应一个聚类中心
            # 如果没有使用转置，即没有.T，那么Q矩阵的每一行代表一个样本的软分配，每一列代表一个聚类中心的分配
            Q = torch.exp(scores / epsilon).t()  # 用指数函数转换分数以获得正值
            # print('Q.shape',Q.shape)  Q.shape torch.Size([2000, 1909])
            Q /= Q.sum(dim=1, keepdim=True)  # 归一化以使每行和为1

            K, B = Q.shape  # K是聚类数量，B是批处理大小
            u = torch.zeros(K).to(scores.device)  # 初始化u和r为0向量
            r = torch.ones(K).to(scores.device) / K  # r是平均分配的目标向量
            c = torch.ones(B).to(scores.device) / B  # c是平均分配的目标向量

            for _ in range(n_iters):
                u = Q.sum(dim=1)  # 按行求和
                Q *= (r / u).unsqueeze(1)  # 更新Q矩阵以满足行约束
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)  # 更新Q矩阵以满足列约束

            return (Q / Q.sum(dim=0, keepdim=True)).t()  # 返回归一化后的Q矩阵

    def train(self):
        model = self.model.cuda()
        batch_size = self.config["batch_size"]
        num_epochs = self.config["num_epochs"]
        lr = self.config["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for index, batch in enumerate(next_batch_pairwise(self.data, batch_size)):
                user_idx, pos_idx, neg_idx = batch

                # brp 损失 LGCN 部分 emb_list 是 all_emb
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb, user_prototypes, item_prototypes  = model()

                # 推荐的brp损失+l2损失
                user_emb, pos_item_emb, neg_item_emb = (
                    rec_user_emb[user_idx],
                    rec_item_emb[pos_idx],
                    rec_item_emb[neg_idx],
                )

                l2_loss = l2_reg_loss(self.config["lambda"], user_emb, pos_item_emb)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_loss

                # Contrastive learning loss
                # Swapping assignments between views loss
                cl_loss = self.cal_cl_loss([user_idx, pos_idx],[rec_user_emb, cl_user_emb], [rec_item_emb, cl_item_emb], [user_prototypes, item_prototypes], self.config["model_config.temperature"])

                batch_loss = rec_loss + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch_loss": batch_loss.item(),
                        "rec_loss": rec_loss.item(),
                        "cl_loss": cl_loss.item(),
                    }
                )

                if index % 100 == 0:
                    print(
                        "training:",
                        epoch + 1,
                        "batch",
                        index,
                        "rec_loss:",
                        rec_loss.item(),
                        "cl_loss",
                        cl_loss.item(),
                    )
            with torch.no_grad():
                self.user_emb, self.item_emb,_,_, _, _ = model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_,_, _, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size,n_layers, prototype_num):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.prototype_num = prototype_num
        self.embedding_dict = self._init_model()
        self.prototypes_dict = self._init_prototypes()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def _init_prototypes(self):
        initializer = nn.init.xavier_uniform_
        prototypes_dict = nn.ParameterDict({
            'user_prototypes': nn.Parameter(initializer(torch.empty(self.prototype_num, self.emb_size))),
            'item_prototypes': nn.Parameter(initializer(torch.empty(self.prototype_num, self.emb_size))),
        })
        return prototypes_dict

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

        user_all_embeddings_cl = all_embeddings[0][:self.data.user_num]
        item_all_embeddings_cl = all_embeddings[0][self.data.user_num:]

        user_prototypes = self.prototypes_dict['user_prototypes']
        item_prototypes = self.prototypes_dict['item_prototypes']
        return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl,user_prototypes,item_prototypes
