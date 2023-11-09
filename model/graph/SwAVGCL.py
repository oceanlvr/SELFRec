import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender

from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import faiss
import wandb
from sklearn.cluster import KMeans

# paper: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW'22
# 为了识别节点 (用户和项目)的语义邻居，HCCF 和 SwAVGCL 追求图结构相邻节点和语义邻居之间的一致表示。


class SwAVGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SwAVGCL, self).__init__(conf, training_set, test_set)
        self.model = LGCN_Encoder(
            self.data,
            self.config["embedding_size"],
            self.config["model_config.eps"],
            self.config["model_config.num_layers"],
        )

    def e_step(self):
        user_embeddings = self.model.embedding_dict["user_emb"].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict["item_emb"].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x"""
        kmeans = faiss.Kmeans(
            d=self.config["embedding_size"],
            k=self.config["model_config.num_clusters"],
            gpu=True,
        )
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def SwaVNCE_loss_1(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(
            initial_emb, [self.data.user_num, self.data.item_num]
        )

        # Normalize embeddings
        user_emb = F.normalize(user_emb, dim=1, p=2)
        item_emb = F.normalize(item_emb, dim=1, p=2)

        # Calculate similarity matrices 计算获得user-user,user-item相似矩阵
        user_user_sim = user_emb[user_idx] @ user_emb.t()
        user_item_sim = user_emb[user_idx] @ item_emb[item_idx].t()

        # Getting the pseudo labels by using the highest similarity
        _, user_pseudo_labels = user_user_sim.max(dim=1)
        _, item_pseudo_labels = user_item_sim.max(dim=1)

        # Calculate cross-entropy loss
        swav_nce_loss_user = F.cross_entropy(user_user_sim, user_pseudo_labels)
        swav_nce_loss_item = F.cross_entropy(user_item_sim, item_pseudo_labels)
        swav_nce_loss = self.config["model_config.proto_reg"] * (
            swav_nce_loss_user + swav_nce_loss_item
        )
        return swav_nce_loss

    def swavloss(self, user_idx, item_idx, temperature=0.1):
        u_idx = torch.unique(torch.Tensor(user_idx).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(item_idx).type(torch.long)).cuda()
        user_view_1, item_view_1, _ = self.model(perturbed=True)
        user_view_2, item_view_2, _ = self.model(perturbed=True)

        # Compute cluster assignment distributions for both views using Sinkhorn-Knopp
        q_user_view_1 = self.sinkhorn_knopp(user_view_1 / temperature)
        q_user_view_2 = self.sinkhorn_knopp(user_view_2 / temperature)
        q_item_view_1 = self.sinkhorn_knopp(item_view_1 / temperature)
        q_item_view_2 = self.sinkhorn_knopp(item_view_2 / temperature)

        # Compute loss - Cross-entropy between soft assignments and features of the other view
        loss_user = -torch.mean(
            torch.sum(
                q_user_view_1 * F.log_softmax(user_view_2 / temperature, dim=1),
                dim=1,
            )
        )
        loss_user += -torch.mean(
            torch.sum(
                q_user_view_2 * F.log_softmax(user_view_1 / temperature, dim=1),
                dim=1,
            )
        )

        loss_item = -torch.mean(
            torch.sum(
                q_item_view_1 * F.log_softmax(item_view_2 / temperature, dim=1),
                dim=1,
            )
        )
        loss_item += -torch.mean(
            torch.sum(
                q_item_view_2 * F.log_softmax(item_view_1 / temperature, dim=1),
                dim=1,
            )
        )

        # 注意这里 proto_nce_loss_item 前面没有去加这个 alpha 系数
        proto_nce_loss = self.config["model_config.proto_reg"] * (loss_user + loss_item)
        return proto_nce_loss

    def sinkhorn_knopp(self, log_Q, num_iters=3, epsilon=1e-3):
        """
        将输入矩阵log_Q转换为双随机形式。

        :param log_Q: 输入矩阵的对数形式。
        :param num_iters: Sinkhorn-Knopp算法迭代次数。
        :param epsilon: 为了数值稳定性而加的小量。
        :return: 调整后的双随机矩阵。
        """
        with torch.no_grad():
            device = log_Q.device  # Get the device from log_Q
            Q = torch.exp(log_Q)  # Convert log probabilities to probabilities.
            sum_Q = torch.sum(Q)
            Q /= sum_Q  # Normalize Q

            r = (
                torch.ones(Q.shape[0], device=device) / Q.shape[0]
            )  # Move r to the device of log_Q
            c = (
                torch.ones(Q.shape[1], device=device) / Q.shape[1]
            )  # Move c to the device of log_Q

            curr_sum = torch.sum(Q, dim=1)

            for _ in range(num_iters):
                # Update rows
                Q *= (r / curr_sum).view(-1, 1)
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= torch.sum(Q, dim=1, keepdim=True)

                # Update columns
                curr_sum = torch.sum(Q, dim=1)
                if torch.max(torch.abs(curr_sum - r)) < epsilon:
                    # Break if the change is smaller than epsilon for numerical stability
                    break

            # Final normalization to avoid numerical issues
            Q *= (r / curr_sum).view(-1, 1)
            Q /= torch.sum(Q, dim=0, keepdim=True)

            return torch.log(Q + epsilon)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning_rate"]
        )
        for epoch in range(self.config["num_epochs"]):
            for index, batch in enumerate(
                next_batch_pairwise(self.data, self.config["batch_size"])
            ):
                user_idx, pos_idx, neg_idx = batch

                # brp 损失 LGCN 部分 emb_list 是 all_emb
                rec_user_emb, rec_item_emb, _ = model()

                # 推荐的brp损失+l2损失
                user_emb, pos_item_emb, neg_item_emb = (
                    rec_user_emb[user_idx],
                    rec_item_emb[pos_idx],
                    rec_item_emb[neg_idx],
                )
                rec_loss = (
                    bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    + l2_reg_loss(
                        self.config["lambda"], user_emb, pos_item_emb, neg_item_emb
                    )
                    / self.config["batch_size"]
                )

                cl_loss = self.swavloss(user_idx, pos_idx)

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
