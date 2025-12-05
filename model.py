from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import GCNConv


def load_model(name, input_dim, hid_dim, output_dim, dropout):
    if name == "gcn":
        return GCN(input_dim, hid_dim, output_dim, dropout)
    elif name == "gat":
        return GAT(input_dim, hid_dim, output_dim, dropout=dropout)


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
    def __init__(self, input_dim, hid_dim, output_dim, dropout, k, a=1.0, b=1.0):
        """
        Args:
            input_dim: 输入特征维度
            hid_dim: 隐藏层维度 (也是多项式滤波特征的维度)
            output_dim: 输出维度 (通常是分类数)
            dropout: Dropout 比率
            k: 雅可比多项式的阶数 (K)
            a, b: 雅可比多项式的超参数 alpha, beta (对应论文中的 a, b)
        """
        super(Jacobi, self).__init__()
        self.k = k
        self.a = a
        self.b = b
        self.dropout = dropout

        # 1. 初步特征提取 (对应论文 MLP 预处理)
        self.mlp = nn.Linear(input_dim, hid_dim)

        # 2. 为每一阶 (0 到 K) 定义一个可训练的权重矩阵 W_k
        # 对应论文 Eq. (13) 中的 W_k
        self.filter_linears = nn.ModuleList([
            nn.Linear(hid_dim, hid_dim) for _ in range(k + 1)
        ])

        # 3. 输出分类层
        self.classifier = nn.Linear(hid_dim, output_dim)

        # 激活函数
        self.act = nn.ReLU()

    def _get_normalized_adj(self, edge_index, num_nodes):
        """
        计算归一化邻接矩阵 A_hat = D^(-1/2) * A * D^(-1/2)
        注意：论文中公式(8)使用的是 g(L_hat) = P(I - L_hat) = P(A_hat)
        所以这里我们需要的是 Normalized Adjacency Matrix。
        """
        # 添加自环 (通常 GCN 需要)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # 计算度矩阵 D
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float,
                          device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(
            row.size(0), device=edge_index.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 构造稀疏矩阵 A_hat
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 为了矩阵乘法方便，这里构建稀疏张量
        shape = torch.Size([num_nodes, num_nodes])
        adj_mat = torch.sparse_coo_tensor(edge_index, values, shape)
        return adj_mat

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]

        # --- 步骤 1: 特征预处理 ---
        x = self.mlp(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(x)  # [N, hid_dim]

        # --- 步骤 2: 准备图结构 (A_hat) ---
        # 这里的 adj 对应公式中的 \hat{A}
        adj = self._get_normalized_adj(edge_index, num_nodes)

        # --- 步骤 3: 雅可比递归 (PolyConv) ---
        # 存储 Z_0 到 Z_K
        Zs = []

        # k=0: Z_0 = X (Eq. 9)
        Z_0 = x
        Zs.append(Z_0)

        if self.k >= 1:
            # k=1: Z_1 = coef1 * X + coef2 * A * X (Eq. 9)
            # coef1 = (a-b)/2, coef2 = (a+b+2)/2
            coef1 = (self.a - self.b) / 2
            coef2 = (self.a + self.b + 2) / 2

            # Sparse Matrix Multiplication: A * X
            AX = torch.sparse.mm(adj, x)
            Z_1 = coef1 * x + coef2 * AX
            Zs.append(Z_1)

        # k >= 2: 递归公式 (Eq. 10 & 7)
        for k_idx in range(2, self.k + 1):
            # 获取前两项
            Z_last = Zs[-1]     # Z_{k-1}
            Z_prev = Zs[-2]     # Z_{k-2}

            # 计算递归系数 (Eq. 7)
            # 注意：论文公式中的 k 是当前的阶数 k_idx
            a, b = self.a, self.b
            k = k_idx

            phi_k = (2*k + a + b) * (2*k + a + b - 1) / (2*k * (k + a + b))
            phi_prime_k = (2*k + a + b - 1) * (a**2 - b**2) / \
                (2*k * (k + a + b) * (2*k + a + b - 2))
            phi_double_prime_k = (k + a - 1) * (k + b - 1) * \
                (2*k + a + b) / (k * (k + a + b) * (2*k + a + b - 2))

            # 计算 Z_k
            # term1 = phi_k * A * Z_{k-1}
            term1 = phi_k * torch.sparse.mm(adj, Z_last)
            # term2 = phi'_k * Z_{k-1}
            term2 = phi_prime_k * Z_last
            # term3 = phi''_k * Z_{k-2}
            term3 = phi_double_prime_k * Z_prev

            Z_next = term1 + term2 - term3
            Zs.append(Z_next)

        # --- 步骤 4: 节点级注意力融合 (Node-level Attention Fusion) ---
        # 对应 SCNAF 模块

        # 预计算所有变换后的特征 H_k = Z_k * W_k
        # Hs 列表包含 K+1 个 [N, hid_dim] 的张量
        Hs = [self.filter_linears[i](Zs[i]) for i in range(self.k + 1)]

        # 计算 Attention Score (w_k) -> Eq. (14)
        # w_k = tanh(H_k * q_k^T)
        # q_k 是 H_k 的均值 (Summary vector)

        scores = []
        for i in range(self.k + 1):
            H_k = Hs[i]  # [N, D]
            # [1, D], 论文中的 vector summary
            q_k = torch.mean(H_k, dim=0, keepdim=True)

            # H_k * q_k^T -> [N, D] * [D, 1] -> [N, 1]
            # 这里的 score 就是论文中的 w_k (未归一化)
            score = torch.tanh(torch.mm(H_k, q_k.t()))
            scores.append(score)

        # 将分数拼接: [N, K+1]
        scores = torch.cat(scores, dim=1)

        # 归一化 (Softmax) -> Eq. (15)
        # alpha: [N, K+1], 每一行代表一个节点对不同阶数的权重分布
        alpha = F.softmax(scores, dim=1)

        # 保存 alpha 以便后续分析或 FGCL 蒸馏使用
        self.last_alpha = alpha.detach()

        # 融合特征 -> Eq. (13)
        # Z_tilde = sum(alpha_k * H_k)
        # 我们可以通过爱因斯坦求和约定或广播来实现

        # Stack Hs: [N, K+1, D]
        Hs_stack = torch.stack(Hs, dim=1)

        # Alpha: [N, K+1] -> [N, K+1, 1]
        alpha_expanded = alpha.unsqueeze(-1)

        # Weighted Sum: [N, D]
        Z_tilde = torch.sum(alpha_expanded * Hs_stack, dim=1)

        # 激活
        Z_tilde = self.act(Z_tilde)

        # --- 步骤 5: 输出 ---
        # 这里的 out 对应论文中只用 SCNAF 的输出，
        # 完整的 CIE-GAD 还需要加超图模块，但作为独立 Jacobi 网络，这里直接输出预测
        out = self.classifier(Z_tilde)

        return out, Z_tilde  # 返回 Z_tilde 方便做 FGCL 的中间层蒸馏
