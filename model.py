from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import GCNConv
from config import args


def load_model(name, input_dim, hid_dim, output_dim, dropout):
    if name == "gcn":
        return GCN(input_dim, hid_dim, output_dim, dropout)
    elif name == "gat":
        return GAT(input_dim, hid_dim, output_dim, dropout=dropout)
    elif name == "jacobi":
        return Jacobi(input_dim, hid_dim, output_dim, dropout, K=args.K)


class GCN(nn.Module):

    def __init__(self, input_dim, hid_dim, output_dim, p=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, output_dim)
        self.p = p

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        embedding = F.dropout(z, p=self.p)
        logits = self.conv2(embedding, edge_index)
        return logits, embedding, None


class GAT(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GATConv(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hid_dim, hid_dim))
            self.layers.append(GATConv(hid_dim, output_dim))
        else:
            self.layers.append(GATConv(input_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        esoft_list = []

        for i, layer in enumerate(self.layers[:-1]):
            x, (edge_index, alpha) = layer(
                x, edge_index, return_attention_weights=True)
            esoft_list.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits, (edge_index, alpha) = self.layers[-1](x,
                                                      edge_index, return_attention_weights=True)
        esoft_list.append(alpha)

        embedding = x
        return logits, embedding, esoft_list

    def decode(self, embedding, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logits = cos(embedding[edge_index[0]], embedding[edge_index[1]])
        logits = (logits+1)/2
        return logits


class Jacobi(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout, K, a=1.0, b=1.0):
        super(Jacobi, self).__init__()
        self.K = K
        self.a = a
        self.b = b
        self.dropout = dropout
        self.hid_dim = hid_dim

        # 1. 初步特征提取
        self.mlp = nn.Linear(input_dim, hid_dim)

        self.W_weight = nn.Parameter(torch.Tensor(
            K + 1, hid_dim, hid_dim))  # W_k
        self.W_bias = nn.Parameter(torch.Tensor(K + 1, hid_dim))

        # 初始化参数 (类似 nn.Linear 的初始化)
        nn.init.kaiming_uniform_(self.W_weight)
        nn.init.zeros_(self.W_bias)

        # 3. 输出分类层
        self.classifier = nn.Linear(hid_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.shape[0]

        # --- 步骤 1: 特征预处理 ---
        x = self.mlp(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=num_nodes, add_self_loops=False)

        A_hat = torch.sparse_coo_tensor(
            indices=edge_index, values=edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        # --- 步骤 3: 雅可比递归 (保持不变) ---
        Z_list = []
        Z_0 = x
        Z_list.append(Z_0)

        if self.K >= 1:
            coef1 = (self.a - self.b) / 2
            coef2 = (self.a + self.b + 2) / 2
            AX = torch.sparse.mm(A_hat, x)
            Z_1 = coef1 * x + coef2 * AX
            Z_list.append(Z_1)

        for k_idx in range(2, self.K + 1):
            Z_last = Z_list[-1]
            Z_prev = Z_list[-2]
            a, b = self.a, self.b
            k = k_idx

            phi_k = (2*k + a + b) * (2*k + a + b - 1) / (2*k * (k + a + b))
            phi_prime_k = (2*k + a + b - 1) * (a**2 - b**2) / \
                (2*k * (k + a + b) * (2*k + a + b - 2))
            phi_double_prime_k = (k + a - 1) * (k + b - 1) * \
                (2*k + a + b) / (k * (k + a + b) * (2*k + a + b - 2))

            term1 = phi_k * torch.sparse.mm(A_hat, Z_last)
            term2 = phi_prime_k * Z_last
            term3 = phi_double_prime_k * Z_prev
            Z_next = term1 + term2 - term3
            Z_list.append(Z_next)

        # --- 【优化点 2】: 并行化特征变换 ---

        # 1. 堆叠 Z_list -> [K+1, N, hid_dim]
        Zs_stack = torch.stack(Z_list, dim=0)

        # 2. 使用 Batch Matrix Multiplication (BMM)
        # Zs_stack:   [K+1, N, hid_dim]
        # W_weight:   [K+1, hid_dim, hid_dim]
        # Hs_stack:   [K+1, N, hid_dim]
        # 相当于同时执行了 K+1 个矩阵乘法
        Hs_stack = torch.bmm(Zs_stack, self.W_weight) + \
            self.W_bias.unsqueeze(1)

        # --- 步骤 4: 节点级注意力融合 (针对优化后的张量调整) ---

        # Hs_stack 已经是 [K+1, N, D]
        # 计算 q_k (Summary vector) -> 在 dim=1 (N) 上取平均
        # q_k: [K+1, 1, D]
        q_k = torch.mean(Hs_stack, dim=1, keepdim=True)

        # 计算 Attention Score
        # Hs_stack: [K+1, N, D]
        # q_k.transpose(1, 2): [K+1, D, 1]
        # bmm 结果: [K+1, N, 1] -> squeeze -> [K+1, N] -> 转置为 [N, K+1]
        scores = torch.bmm(Hs_stack, q_k.transpose(1, 2)).squeeze(-1).t()

        scores = torch.tanh(scores)  # [N, K+1]

        # 归一化
        alpha = F.softmax(scores, dim=1)  # [N, K+1]

        # 融合特征
        # alpha: [N, K+1] -> [N, K+1, 1]
        # Hs_stack 需要转置一下以匹配 alpha 的维度用于广播:
        # 目前 Hs_stack 是 [K+1, N, D]，我们需要 [N, K+1, D]
        Hs_stack_permuted = Hs_stack.permute(1, 0, 2)

        # Weighted Sum
        # [N, K+1, 1] * [N, K+1, D] -> sum over K+1 (dim=1) -> [N, D]
        Z_tilde = torch.sum(alpha.unsqueeze(-1) * Hs_stack_permuted, dim=1)

        Z_tilde = self.act(Z_tilde)
        Z_tilde = F.dropout(Z_tilde, p=self.dropout)

        out = self.classifier(Z_tilde)

        # 注意：这里返回的 W 需要改为 self.W_weight 或转回 List 形式，取决于外部如何使用
        return out, Z_tilde, Zs_stack, alpha
