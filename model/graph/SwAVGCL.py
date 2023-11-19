import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender

from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import wandb


class SwAVGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SwAVGCL, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config["embedding_size"],
            self.config["model_config.eps"],
            self.config["model_config.num_layers"],
        )
        self.user_prototypes = nn.Parameter(
            torch.randn(
                self.config["embedding_size"],
                self.config["model_config.num_clusters"],
            )
        )
        self.item_prototypes = nn.Parameter(
            torch.randn(
                self.config["embedding_size"],
                self.config["model_config.num_clusters"],
            )
        )

    def cal_cl_loss(self, idx):
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        user_z = torch.concat(user_view_1, user_view_2)
        item_z = torch.concat(item_view_1, item_view_2)

        user_loss = self.swav_loss(user_z, self.user_prototypes)
        item_loss = self.swav_loss(item_z, self.item_prototypes)
        cl_loss = user_loss + item_loss

        # # 获取用户和物品的嵌入
        # user_emb = user_embeddings[user_idx]
        # item_emb = item_embeddings[item_idx]
        # # 计算用户和物品的原型激活值
        # user_proto_activations = torch.matmul(user_emb, self.user_prototypes)
        # item_proto_activations = torch.matmul(item_emb, self.item_prototypes)
        return cl_loss

    def swav_loss(self, z, prototypes, temperature=0.1):
        # Compute scores between embeddings and prototypes
        scores = torch.mm(z, prototypes)

        score_t = scores[: z.size(0) // 2]
        score_s = scores[z.size(0) // 2 :]

        # Apply the Sinkhorn-Knopp algorithm to get soft cluster assignments
        q_t = self.sinkhorn_knopp(score_t)
        q_s = self.sinkhorn_knopp(score_s)

        log_p_t = torch.log_softmax(score_t / temperature, dim=1)
        log_p_s = torch.log_softmax(score_s / temperature, dim=1)

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
        swav_loss = -((loss_t + loss_s) / 2)
        return swav_loss

    def sinkhorn_knopp(self, scores, epsilon=0.05, n_iters=3):
        with torch.no_grad():
            # 我们通常希望返回的矩阵Q的每一列代表一个样本的软聚类分配，而每一行对应一个聚类中心
            Q = torch.exp(scores / epsilon).t()  # 用指数函数转换分数以获得正值
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
                rec_user_emb, rec_item_emb, _ = model()

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
                cl_loss = self.cal_cl_loss(
                    rec_user_emb, rec_item_emb, user_idx, pos_idx
                )

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
    def __init__(self, data, emb_size, eps, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(
            self.norm_adj
        ).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.data.user_num, self.emb_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.data.item_num, self.emb_size))
                ),
            }
        )
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += (
                    torch.sign(ego_embeddings)
                    * F.normalize(random_noise, dim=-1)
                    * self.eps
                )
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.data.user_num, self.data.item_num]
        )
        return user_all_embeddings, item_all_embeddings, all_embeddings
