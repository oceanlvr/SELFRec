import numpy as np
import scipy.sparse as sp

class Graph(object):
    def __init__(self):
        pass
    # D^(-1/2)·A·D^(-1/2) 是图邻接矩阵A的normaliztion表示。
    # 图的正则化矩阵
    # 理论基础是 https://blog.csdn.net/mzy20010420/article/details/127557236
    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1)) # 拿到矩阵的度向量 D
        if shape[0] == shape[1]:    
            d_inv = np.power(rowsum, -0.5).flatten() # 1/sqrt(度向量) d^(-1/2)
            d_inv[np.isinf(d_inv)] = 0.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
            # diags https://blog.csdn.net/mercies/article/details/108513787
            # diagonal 是对矩阵对角化的意思
            # 这里是构造了 D^(-1/2) <-> d_mat_inv, A <-> adj_mat
            d_mat_inv = sp.diags(d_inv) # d->D Construct a sparse matrix from diagonals. 从对角线构建一个稀疏矩阵。
            norm_adj_tmp = d_mat_inv.dot(adj_mat) # D^(-1/2)*A
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv) # D^(-1/2)*A*D^(-1/2)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat) # D^(-1/2)*A
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass
