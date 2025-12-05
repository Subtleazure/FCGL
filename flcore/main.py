# main.py

import torch
from generator import GraphGenerator, prob_matrix_to_edge_index
from torch_geometric.data import Data

def main():
    print("start")
    
    # 设置参数
    nz = 100              # 噪声维度
    node_feat_dim = 128   # 节点特征维度
    hidden_dim = 256      # 隐藏层维度
    num_nodes = 1000       # 图中节点数量

    # 初始化生成器
    generator = GraphGenerator(
        nz=nz,
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes
    )

    # 使用 GPU 如果可用
    num = 5
    device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
    generator.to(device)

    # 生成随机噪声
    z = torch.randn(1, nz).to(device)  # batch_size=1

    # 前向传播，生成节点特征和边概率矩阵
    with torch.no_grad():
        node_features, edge_index = generator(z)
    print("node_features:", node_features.shape)  # [1, 20, 128]
    print("edge_index:", edge_index.shape)       # [1, 20, 20]

    feat = node_features[0]  # [20, 128]
    print("feat:", feat.shape)
    save_edge_index_to_txt(edge_index, "edge_index.txt")
    graph_data = Data(x=feat, edge_index=edge_index, num_nodes=num_nodes)

        # 可以在这里打印或保存 graph_data，例如：
        # print(f'Graph {i}: {graph_data}')

    print("ok")

def save_edge_index_to_txt(edge_index, filename):
    # 转置成 [E, 2]，每一行是一个 (src, dst) 边
    edges = edge_index.t().cpu().numpy()  # 转为 numpy，便于处理

    with open(filename, 'w') as f:
        for src, dst in edges:
            f.write(f"{src}, {dst}\n")
    print(f"边列表已保存到 {filename}")

if __name__ == "__main__":
    main()