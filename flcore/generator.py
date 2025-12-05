import torch
import torch.nn as nn
from torch_geometric.data import Data

class GraphGenerator(nn.Module):
    """图生成器：生成节点特征和边结构"""
    def __init__(self, nz=100, node_feat_dim=128, hidden_dim=256, num_nodes=1000, k=5000):
        super(GraphGenerator, self).__init__()
        self.nz = nz
        self.node_feat_dim = node_feat_dim
        self.num_nodes = num_nodes
        self.k = k
        self.pos_emb = nn.Embedding(self.num_nodes, self.node_feat_dim)
        
        # 在 __init__ 中
        self.node_feature_net = nn.Sequential(
            nn.Linear(nz + node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
            nn.Tanh()
        )
        
        # 自注意力层用于节点间关系
        self.attention = nn.MultiheadAttention(
            embed_dim=node_feat_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 边预测MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)  # z: [B, nz]

        pos_emb = self.pos_emb(torch.arange(self.num_nodes, device=z.device))  # [N, F]
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, F]

        # 与 z 扩展后的特征融合
        z_expanded = z.unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B, N, nz]
        combined = torch.cat([z_expanded, pos_emb], dim=-1)  # [B, N, nz + F]
        node_features = self.node_feature_net(combined)       # [B, N, 128]  # [B, N, F]
        # 自注意力
        attended_features, _ = self.attention(node_features, node_features, node_features)
        node_features = attended_features  # [B, N, F]

        # 构造所有节点对的特征拼接
        # node_i: [B, N, 1, F] -> expand -> [B, N, N, F]
        # node_j: [B, 1, N, F] -> expand -> [B, N, N, F]
        node_i = node_features.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        node_j = node_features.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)

        # 拼接特征
        pair_features = torch.cat([node_i, node_j], dim=-1)  # [B, N, N, 2F]

        # 预测边概率
        edge_probs = self.edge_mlp(pair_features).squeeze(-1)  # [B, N, N]

        edge_index = prob_matrix_to_edge_index(edge_probs[0], k = self.k)

        return node_features, edge_index
    
def prob_matrix_to_edge_index(prob_matrix, k=5000, remove_self_loops=True, to_undirected=True):
    """
    将概率矩阵转为 edge_index，支持去自环、转无向图，并选择概率最高的 k 条边。
    :param prob_matrix: [N, N] 概率矩阵
    :param k: 需要选择的边的数量
    :param remove_self_loops: 是否去除自环
    :param to_undirected: 是否转为无向图
    :return: edge_index [2, E]
    """
    N = prob_matrix.size(0)
    
    # 1. 移除自环
    if remove_self_loops:
        prob_matrix = prob_matrix * (1 - torch.eye(N, device=prob_matrix.device))
    
    # 2. 如果需要转换为无向图，则只考虑上三角部分
    if to_undirected:
        upper_tri_mask = torch.triu(torch.ones((N, N), dtype=torch.bool, device=prob_matrix.device), diagonal=1)
        prob_matrix_upper = prob_matrix.masked_select(upper_tri_mask).flatten()
        
        # 3. 执行 topk 操作
        if k > len(prob_matrix_upper):
            k = len(prob_matrix_upper)  # 如果 k 大于可选边数，则选择所有可用边
        topk_values, topk_indices = torch.topk(prob_matrix_upper, k)
        
        # 4. 转换回原始索引
        row_upper, col_upper = torch.where(upper_tri_mask)
        selected_rows = row_upper[topk_indices]
        selected_cols = col_upper[topk_indices]
        
        # 构建无向图的 edge_index
        edge_index = torch.stack([selected_rows, selected_cols], dim=0)
        
        # 对于无向图，还需要添加反方向的边
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        # 如果不需要转换为无向图，则直接操作整个矩阵
        flat_probs = prob_matrix.flatten()
        if k > len(flat_probs):
            k = len(flat_probs)
        topk_values, topk_indices = torch.topk(flat_probs, k)
        
        row = topk_indices // N
        col = topk_indices % N
        edge_index = torch.stack([row, col], dim=0)

    return edge_index

class ServerGraphSynthesizer:
    """服务器端图合成器"""
    def __init__(self, global_model, generator, nz=100, num_classes=10, 
                 synthesis_batch_size=256, iterations=50, lr_g=0.001):
        self.global_model = global_model
        self.generator = generator
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.ce_loss = nn.CrossEntropyLoss()
        self.synthesis_batch_size = synthesis_batch_size
        self.iterations = iterations
        self.nz = nz
        self.num_classes = num_classes
    
    def synthesize(self):
        """生成合成图数据"""
        self.global_model.eval()
        self.generator.train()
        best_graphs = []
        
        for iteration in range(self.iterations):
            z = torch.randn(self.synthesis_batch_size, self.nz).cuda()
            targets = torch.randint(0, self.num_classes, (self.synthesis_batch_size,)).cuda()
            
            node_features, edge_probs = self.generator(z)
            
            synthetic_graphs = []
            total_loss = 0
            
            for i in range(self.synthesis_batch_size):
                edge_index = prob_matrix_to_edge_index(edge_probs[i])
                graph_data = Data(x=node_features[i].unsqueeze(0), edge_index=edge_index, y=targets[i].unsqueeze(0))
                
                with torch.no_grad():
                    logits, _, _ = self.global_model(graph_data)
                
                loss = self.ce_loss(logits, targets[i].unsqueeze(0))
                total_loss += loss
                
                if iteration == self.iterations - 1:  # 最后一次迭代保存
                    synthetic_graphs.append(graph_data)
            
            avg_loss = total_loss / self.synthesis_batch_size
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()
            
            if iteration == self.iterations - 1:
                best_graphs = synthetic_graphs
        
        return best_graphs