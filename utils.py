import copy
import random
import subprocess
import sys
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
import torch
from torch_geometric.utils import coalesce, dense_to_sparse, add_self_loops, remove_self_loops
import os
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch.nn import init
from torch.utils.data import Dataset


class GraphGenerator(nn.Module):
    def __init__(self, nz=128, hidden_dim=256, feat_dim=64, num_classes=7, max_nodes=128, temperature=1.0):
        super(GraphGenerator, self).__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.temperature = temperature

        # Latent to hidden
        self.z_to_hidden = nn.Sequential(
            nn.Linear(nz, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Generate node features
        self.feat_mlp = nn.Sequential(
            nn.Linear(nz, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Tanh()
        )

        # Attention-based edge generator
        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.edge_threshold = nn.Parameter(torch.tensor(0.5))  # 可学习阈值

        # Node number predictor
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, max_nodes),
            nn.Sigmoid()
        )

    def forward(self, z, label_distribution='random'):
        """
        输入噪声 z，输出一个图 Batch，每个节点有 x, y（标签）

        Args:
            z: [B, nz]
            label_distribution: 'random', 'uniform', 'balanced'

        Returns:
            Batch of Data(x, edge_index, y, num_nodes)
        """
        h = self.z_to_hidden(z)  # [B, H]
        B = z.size(0)

        # Predict number of nodes per graph
        n_nodes_logits = self.node_predictor(h)  # [B, max_nodes]
        n_nodes = 2 + torch.argmax(n_nodes_logits, dim=1)  # 至少2个节点
        max_n = n_nodes.max().item()

        # Expand h to node-level
        h_repeated = h.unsqueeze(1).repeat(1, max_n, 1)  # [B, max_n, H]

        # Generate node features
        x = self.feat_mlp(h_repeated)  # [B, max_n, feat_dim]

        # Use attention to compute edge scores
        x_for_attn = x  # [B, N, feat_dim]
        attn_weights, _ = self.attention(
            x_for_attn, x_for_attn, x_for_attn)  # [B, N, N]
        attn_weights = torch.sigmoid(
            attn_weights.mean(dim=1))  # 平均多头 [B, N, N]

        # Apply threshold to get sparse edges
        threshold = torch.sigmoid(self.edge_threshold)
        edge_mask = (attn_weights > threshold).float()

        # Build edge_index for each graph in batch
        data_list = []
        for i in range(B):
            num_n = n_nodes[i].item()
            x_i = x[i, :num_n]  # [num_n, feat_dim]

            if label_distribution == 'random':
                y_i = torch.randint(0, self.num_classes, (num_n,))
            elif label_distribution == 'balanced':
                # 尽可能平衡分配
                class_ids = np.array(
                    [i % self.num_classes for i in range(num_n)])
                np.random.shuffle(class_ids)
                y_i = torch.tensor(class_ids, dtype=torch.long)
            else:
                raise ValueError(
                    "label_distribution must be 'random' or 'balanced'")

            # Extract edges
            adj_i = edge_mask[i, :num_n, :num_n]
            edge_index = (adj_i > 0).nonzero(
                as_tuple=False).t().contiguous()  # [2, E]

            data_list.append(
                Data(x=x_i, edge_index=edge_index, y=y_i, num_nodes=num_n))

        return Batch.from_data_list(data_list)


class GraphSynthesizer:
    def __init__(self, args, teacher, student, generator, iterations=100, nz=128,
                 synthesis_batch_size=8, lr_g=0.002, lr_z=0.01,
                 kd_T=2.0, warmup=10, save_dir="graph_syn", device="cuda"):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device).train()
        self.generator = generator.to(device).train()
        self.nz = nz
        self.num_classes = generator.num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.kd_T = kd_T
        self.warmup = warmup
        self.device = device
        self.save_dir = save_dir
        self.ep = 0
        self.args = args
        self.iterations = iterations
        # Optimizer for generator and latent codes
        self.meta_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g * 100, betas=(0.5, 0.999))

    def synthesize(self, targets=None):
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_loss = float('inf')
        best_graphs = None

        # Latent code
        z = torch.randn(self.synthesis_batch_size, self.nz,
                        requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters(), 'lr': self.lr_g},
            {'params': [z], 'lr': self.lr_z}
        ], betas=(0.5, 0.999))

        for step in range(self.iterations):  # 内循环优化 z 和 generator
            optimizer.zero_grad()

            # 生成图 batch
            synth_graphs = self.generator(z)
            if hasattr(synth_graphs, 'batch'):
                batch = synth_graphs.batch
            else:
                batch = None

            if targets is None:
                # 使用教师预测作为软标签
                with torch.no_grad():
                    t_out = self.teacher(synth_graphs)
                    t_logits = t_out["logits"]
                    soft_labels = F.softmax(t_logits / self.kd_T, dim=1)
            else:
                soft_labels = F.one_hot(
                    targets, num_classes=self.num_classes).float().to(self.device)

            # 学生模型推理
            s_out = self.student(synth_graphs)
            s_logits = s_out["logits"]

            # 蒸馏损失
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / self.kd_T, dim=1),
                soft_labels,
                reduction='batchmean'
            ) * (self.kd_T ** 2)

            # 分类损失（可选）
            loss_cls = F.cross_entropy(t_logits, soft_labels.argmax(dim=1))

            # 组合损失
            loss = loss_kd + 0.5 * loss_cls

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_graphs = synth_graphs.clone().to(
                    f'cuda:{self.args.gpuid}')

        # REPTILE-style 更新 generator
        if self.ep >= self.warmup:
            self.meta_optimizer.zero_grad()
            # 梯度方向：原始 generator -> 优化后的 generator
            for p_src, p_tgt in zip(self.generator.parameters(), self.generator.parameters()):
                if p_src.grad is None:
                    p_src.grad = (p_src.data - p_tgt.data).detach().clone()
            self.meta_optimizer.step()

        # 保存到池
        # self.data_pool.append(best_graphs)
        return best_graphs


def kd_train(student, teacher, graph_batch_iter, criterion, optimizer, device, epochs=200):
    student.train()
    teacher.eval()
    for epoch in range(epochs):
        graphs = graph_batch_iter.next().to(device)

        with torch.no_grad():
            t_out = teacher(graphs)
            t_logits = t_out["logits"].detach()

        s_out = student(graphs)
        s_logits = s_out["logits"]

        loss = criterion(s_logits, t_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T*T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def gcn_weight_init(m):
    init.xavier_normal_(m.lin.weight.data)
    if m.lin.bias is not None:
        init.constant_(m.lin.bias.data, 0)


class GraphDataset(Dataset):
    def __init__(self, graph_list):
        """
        graph_list: list of batched graph objects:
        [batched_graph1, batched_graph2, ...]
        """
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data
#########################


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def start(args, fcgl_dataset, clients, server, message_pool, device):
    if args.global_eval:
        assert not args.isolate_mode

    # data analysis
    print("-"*50)
    for client_id in range(args.num_clients):
        result_list = []
        for task_id in range(len(clients[client_id].data["task"])):
            label = clients[client_id].data["data"].y
            task = clients[client_id].data["task"][task_id]
            node_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
            result = torch.unique(label[node_mask]).cpu().tolist()
            result_list.append(str(result))
        print(f"client {client_id} tasks: {'=>'.join(result_list)}")

    # start train
    print("-"*50)
    num_tasks = len(fcgl_dataset[0]["task"])
    args.num_tasks = num_tasks

    FL_acc_matrix = torch.zeros(size=(num_tasks, num_tasks)).to(device)

    for task_id in range(num_tasks):
        FL_acc_matrix[task_id, :] = 0
        message_pool["task_id"] = task_id
        for client_id in range(args.num_clients):
            clients[client_id].task_start(task_id)
        server.task_start(task_id)
        for round_id in range(args.num_rounds_per_task):
            message_pool["round_id"] = round_id
            server.send_message()
            for client_id in range(args.num_clients):
                clients[client_id].execute(task_id)
                clients[client_id].send_message(task_id)
            server.execute()
        # ↑ 训练  |  评估 ↓
        for eval_task_id in range(0, task_id+1):
            total_nodes = 0
            for client_id in range(args.num_clients):
                client_acc = clients[client_id].evaluate(
                    task_id=eval_task_id, use_global=args.global_eval)
                num_nodes = clients[client_id].data["task"][eval_task_id]["test_mask"].sum(
                )
                FL_acc_matrix[task_id, eval_task_id] += client_acc * num_nodes
                total_nodes += num_nodes
            FL_acc_matrix[task_id, eval_task_id] /= total_nodes
            print(
                f"[Task {task_id} Finish] Global Accuracy on Task {eval_task_id}: {FL_acc_matrix[task_id, eval_task_id]:.2f}")
        print(FL_acc_matrix)
        aa = AA(FL_acc_matrix, T=task_id+1)
        af = AF(FL_acc_matrix, T=task_id+1)
        print(
            f"[Task {task_id} Finish] Global AA: {aa:.2f}\tGlobal AF: {af:.2f}")
        for client_id in range(args.num_clients):
            clients[client_id].task_done(task_id)
        # add M
        if args.model == "jacobi":
            for client_id in range(args.num_clients):
                client = clients[client_id]
                eval_model = client.local_model.eval()
                whole_data = client.data["data"].to(client.device)
                task = client.data["task"][task_id]
                task_data = client.task_data(
                    task_id, whole_data, task)
                with torch.no_grad():
                    _, _, Zs_stack, alpha = eval_model(task_data)
                update_memory_with_sgp(args, client, task_id, Zs_stack, alpha)
        server.task_done(task_id)
        print("-"*50)

    return aa, af


def load_clients_server(args, fcgl_dataset, device):
    message_pool = {}

    if args.method == "ours":
        from flcore.Ours import OursClient, OursServer
        clients = [OursClient(args, client_id, fcgl_dataset[client_id],
                              message_pool, device) for client_id in range(args.num_clients)]
        server = OursServer(args, message_pool, device)

    return clients, server, message_pool


def edge_masking(edge_index, handled, device):
    num_nodes = edge_index.max().item()+1
    node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    for node in handled:
        node_mask[node] = True
    mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, mask].to(device)
    self_loop_indices = torch.tensor(
        [[node, node] for node in handled], dtype=torch.long).t().to(device)
    edge_index = torch.cat([edge_index, self_loop_indices], dim=1).to(device)
    edge_index = coalesce(edge_index)
    return edge_index


def accuracy(logits, label):
    pred = logits.max(1)[1]
    correct = (pred == label).sum()
    total = label.shape[0]
    return (correct / total)*100


def AA(M_acc, T=None):
    if T is None:
        T = M_acc.size(0)
    result = 0
    for i in range(0, T):
        result += M_acc[T-1, i]
    result /= T
    return result


def AF(M_acc, T=None):
    if T is None:
        T = M_acc.size(0)
    if T == 1:  # single task
        return -1  # error
    result = 0
    for i in range(0, T-1):
        forgetting = M_acc[i, i] - M_acc[T-1, i]
        result += forgetting
    result /= T-1
    return result


# def update_buffer(buffer, replay, task, task_data, embedding, num_samples_per_class):
#     if num_samples_per_class == 0:
#         return buffer


#     if replay == "CM":
#         new_buffer = update_converage_max(buffer, task, task_data, embedding, num_samples_per_class)

#     return new_buffer


def isolate_graph(x, y):
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index.to(x.device), y=y)


def update_random(buffer, task, task_data, num_samples_per_class):
    train_mask = task["train_mask"]
    train_mask = train_mask.to(task_data.y.device)
    all_classes = torch.unique(task_data.y[train_mask])
    x = []
    y = []
    for class_i in all_classes:
        class_i_train_mask = train_mask & (task_data.y == class_i)
        class_i_train_list = class_i_train_mask.nonzero().squeeze().tolist()
        if type(class_i_train_list) is not list:
            class_i_train_list = [class_i_train_list]
        selected_nodes = random.sample(
            class_i_train_list, num_samples_per_class)
        x.append(task_data.x[selected_nodes])
        y.append(task_data.y[selected_nodes])

    added_x = torch.cat(x)
    added_y = torch.cat(y)

    if buffer["x"] is None:
        new_buffer = {"x": added_x, "y": added_y}
    else:
        new_buffer = {"x": torch.cat(
            (buffer["x"], added_x)), "y": torch.cat((buffer["y"], added_y))}
    return new_buffer


def update_converage_max(buffer, task, task_data, embedding, num_samples_per_class, distance=0.1):
    replay = {}
    train_mask = task["train_mask"]
    train_mask = train_mask.to(task_data.y.device)
    all_classes = torch.unique(task_data.y[train_mask])
    x = []
    y = []

    replay_buffer = None

    for class_i in all_classes:
        class_i_train_mask = train_mask & (task_data.y == class_i)
        class_i_train_list = class_i_train_mask.nonzero().squeeze().tolist()
        if type(class_i_train_list) is not list:
            class_i_train_list = [class_i_train_list]

        count = 0
        memory = []
        temp_memory = []
        cover = []

        dist_matrix = torch.cdist(
            embedding[class_i_train_mask], embedding[class_i_train_mask], p=2)
        distances_mean = torch.mean(dist_matrix)
        dist_bin_matrix = torch.where(
            dist_matrix < distances_mean.item() * distance, 1, 0)

        temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
        while count < num_samples_per_class:
            ind = (torch.sum(temp_dist_bin_matrix, 0)).argmax()
            memory.append(class_i_train_list[ind])
            temp_memory.append(ind)
            target_dist_matrix = temp_dist_bin_matrix[ind, :]
            new_cover = torch.where(target_dist_matrix == 1)[0]
            cover = list(set(cover) | set(new_cover))
            temp_dist_bin_matrix[new_cover, :] = 0
            temp_dist_bin_matrix[:, new_cover] = 0

            # reset
            if len(cover) >= len(class_i_train_list) * 0.9:
                cover = temp_memory
                temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
                temp_dist_bin_matrix[cover, :] = 0
                temp_dist_bin_matrix[:, cover] = 0

            count += 1

        memory = torch.from_numpy(np.array(memory))

        replay[class_i] = memory

        replay_buffer = replay[class_i][:num_samples_per_class]
        x.append(task_data.x[replay_buffer])
        y.append(task_data.y[replay_buffer])

    added_x = torch.cat(x)
    added_y = torch.cat(y)

    if buffer["x"] is None:
        new_buffer = {"x": added_x, "y": added_y}
    else:
        new_buffer = {"x": torch.cat(
            (buffer["x"], added_x)), "y": torch.cat((buffer["y"], added_y))}
    return new_buffer


def run_subprocess(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='')
            sys.stdout.flush()

    sys.exit(0)


def construct_self_loop_graph(x, y=None):
    edge_index = torch.empty((2, 0), dtype=torch.long).to(x.device)
    if y is not None:
        return Data(x=x, edge_index=edge_index, y=y.to(x.device))
    else:
        return Data(x=x, edge_index=edge_index)


def construct_knn_graph(x, k=1):
    adj_logits = F.sigmoid(torch.mm(x, x.T))
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(x.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = remove_self_loops(edge_index)[0]
    data = Data(x=x, edge_index=edge_index)

    return data


def check_client_data(args, clients):
    label = []
    counter = []
    for client_id in range(args.num_clients):
        client = clients[client_id]
        for task_id in range(len(client.data['task'])):
            task = client.data["task"][task_id]
            whole_data = client.data["data"]
            selected_index = task["train_mask"] | task["val_mask"] | task["test_mask"]
            task_y = whole_data.y[selected_index]
            for i in torch.unique(task_y):
                label.append(i.item())
                counter.append((task_y == i).sum().item())
                print(
                    f"client {client_id} task {task_id}: class {i.item()} = {(task_y == i).sum().item()}")

    print(label)
    print(counter)


def update_memory_with_sgp(args, client, task_id, Zs_stack, alpha):
    """
    更新客户端的频谱基底记忆 (M) 和重要性 (lam)。

    参数:
    - args: 包含超参数 (batch_size, threshold, gamma/alpha参数, hid_dim)
    - client: 当前客户端对象，存储了 M, lam
    - task_id: 当前任务ID
    - Zs_stack: [K+1, N, D] 所有阶的输入特征 (CPU Tensor)
    - alpha: [N, K+1] Attention权重 (CPU Tensor)
    """

    # 获取超参数
    epsilon = args.epsilon   # 能量阈值 (e.g., 0.95 - 0.99)
    gamma = args.gamma         # SGP的重要性缩放系数 (e.g., 100 or larger)
    batch_size = args.batch_size
    device = client.device     # 使用客户端的GPU进行加速

    K = Zs_stack.shape[0] - 1
    hid_dim = Zs_stack.shape[2]

    # # 初始化存储 (如果是第一个任务)
    # if not hasattr(client, 'M') or client.M is None:
    #     client.M = {k: None for k in range(K + 1)}
    #     client.lam = {k: None for k in range(K + 1)}

    # --- 针对每一阶 K 分别处理 ---
    for k in range(K + 1):
        # 1. 准备数据
        # Z_k: [N, D]
        Z_k = Zs_stack[k]
        num_nodes = Z_k.shape[0]

        # 计算当前任务对第 k 阶的平均偏好 (用于最终重要性加权)
        # alpha: [N, K+1] -> alpha[:, k] -> scalar
        avg_alpha = alpha[:, k].mean().item()

        # 2. 计算全图协方差矩阵 C = Z^T * Z (分块计算以节省显存)
        # C 只有 [D, D] 大小，非常小
        Cov_total = torch.zeros((hid_dim, hid_dim), device=device)

        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                # 切片并移入GPU
                batch_Z = Z_k[i: i + batch_size].to(device)
                # 累加协方差: [D, B] @ [B, D] -> [D, D]
                Cov_total += torch.matmul(batch_Z.t(), batch_Z)

        # 计算总能量 (Trace of Covariance)
        total_energy = torch.trace(Cov_total)

        # 3. 获取旧基底
        M_old = client.M[k]     # [D, r_old] (Tensor on CPU or GPU)
        lam_old = client.lam[k]  # [r_old]

        if M_old is not None:
            M_old = M_old.to(device)
            lam_old = lam_old.to(device)

            # --- SGP 核心逻辑: 计算旧基底在当前数据下的代理奇异值 ---

            # 投影协方差 C_proj = M^T * C * M
            # 物理含义: 当前数据在旧基底空间内的能量分布
            Cov_proj = torch.mm(M_old.t(), torch.mm(Cov_total, M_old))

            sigmas_proxy_sq = torch.diagonal(Cov_proj)
            # 得到 SGP 论文中的 sigma' (surrogate singular values)
            sigmas_proxy = torch.sqrt(torch.abs(sigmas_proxy_sq))

            # --- 更新旧基底的重要性 (Max-Pooling) ---

            # SGP重要性公式: lambda = (gamma + 1) * sig / (gamma * sig + max_sig)
            # 注意: 这里的 max_sig 通常取当前分布的最大值，这里取 sigmas_proxy 的最大值
            # max_sigma_proxy = sigmas_proxy[0] if len(sigmas_proxy) > 0 else 1.0

            # lam_proxy_raw = ((gamma + 1) * sigmas_proxy) / \
            #     (gamma * sigmas_proxy + max_sigma_proxy + 1e-8)

            # # 结合任务偏好 avg_alpha
            # lam_proxy_weighted = lam_proxy_raw * avg_alpha

            # # 核心: 取历史最大值 (防遗忘)
            # # 假设 lam_old 和 lam_proxy 维度一致 (基底固定顺序)
            # lam_old_updated = torch.max(lam_old, lam_proxy_weighted)

            # --- 计算新基底 (Residual) ---

            # 我们需要找 Z 在 M_old 正交补空间上的成分
            # 协方差公式: C_res = (I - MM^T) C (I - MM^T)
            # 展开后: C_res = C - M M^T C - C M M^T + M (M^T C M) M^T
            # 利用已计算的中间变量加速

            MMT_C = torch.mm(torch.mm(M_old, M_old.t()), Cov_total)
            term_last = torch.mm(M_old, torch.mm(
                Cov_proj, M_old.t()))  # M * (M^T C M) * M^T

            Cov_res = Cov_total - MMT_C - MMT_C.t() + term_last

            # 确保对称性 (消除数值误差)
            Cov_res = (Cov_res + Cov_res.t()) / 2

            # 特征分解残差协方差
            eigvals_res, V_res = torch.linalg.eigh(Cov_res)

            # 排序
            idx = torch.argsort(eigvals_res, descending=True)
            eigvals_res = eigvals_res[idx]
            V_res = V_res[:, idx]

            # 能量截断，确定新基底数量
            energy_old = torch.sum(torch.square(sigmas_proxy))  # 注意这里要用平方和作为能量
            current_energy = energy_old
            target_energy = epsilon * total_energy  # right

            num_new_bases = 0
            for i in range(len(eigvals_res)):
                if current_energy >= target_energy:
                    break
                if eigvals_res[i] > 0:
                    current_energy += eigvals_res[i]
                    num_new_bases += 1

            # 提取新基底及其奇异值
            if num_new_bases > 0:
                M_new = V_res[:, :num_new_bases]
                sigmas_new = torch.sqrt(eigvals_res[:num_new_bases])  # [r_new]
            else:
                M_new = torch.empty((hid_dim, 0), device=device)
                sigmas_new = torch.tensor([], device=device)

            # =======================================================
            # 【核心修改】：联合归一化 (Joint Normalization)
            # =======================================================

            # 1. 拼接所有的奇异值 (旧代理 + 新残差)
            all_sigmas = torch.cat([sigmas_proxy, sigmas_new])

            if len(all_sigmas) > 0:
                # 2. 找到当前任务视角的全局最大能量
                max_sigma_total = torch.max(all_sigmas)

                # 3. 统一计算 lambda (SGP Eq. 2)
                # 这里的 lam_all 是当前任务对所有基底(新+旧)的“当前评价”
                lam_all_raw = ((gamma + 1) * all_sigmas) / \
                    (gamma * all_sigmas + max_sigma_total)

                # 4. 乘以任务偏好 alpha
                # lam_all_weighted = lam_all_raw * avg_alpha
                lam_all_weighted = lam_all_raw
                # 5. 拆分回旧部分和新部分
                r_old = len(sigmas_proxy)
                lam_proxy_weighted = lam_all_weighted[:r_old]
                lam_new_weighted = lam_all_weighted[r_old:]

                # 6. 历史更新 (Max-Pooling)
                # 即使 lam_proxy_weighted 因为 max_sigma_total 变大而变小了，
                # lam_old 里的历史高分依然会被保留。
                lam_old_updated = torch.max(lam_old, lam_proxy_weighted)

                # 7. 拼接最终结果
                M_final = torch.cat([M_old, M_new], dim=1)
                lam_final = torch.cat([lam_old_updated, lam_new_weighted])

            else:
                # 极端情况：没有能量
                M_final = M_old
                lam_final = lam_old
        else:
            # --- 第一次任务: 直接对 Cov_total 做分解 ---
            eigvals, V = torch.linalg.eigh(Cov_total)

            # 排序
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            V = V[:, idx]

            # 阈值截断
            current_energy = 0
            target_energy = epsilon * total_energy
            num_bases = 0

            for i in range(len(eigvals)):
                if current_energy >= target_energy:
                    break
                current_energy += eigvals[i]
                num_bases += 1

            M_final = V[:, :num_bases]
            sigmas = torch.sqrt(eigvals[:num_bases])

            # 计算重要性
            if num_bases > 0:
                max_sigma = sigmas[0]
                lam_raw = ((gamma + 1) * sigmas) / \
                    (gamma * sigmas + max_sigma + 1e-8)
                # lam_final = lam_raw * avg_alpha
                lam_final = lam_raw
            else:
                # 极端情况处理
                lam_final = torch.tensor([], device=device)

        # 4. 保存回客户端 (存回 CPU 以节省显存)
        client.M[k] = M_final.cpu()
        client.lam[k] = lam_final.cpu()

        # 显存清理
        del Cov_total
        if M_old is not None:
            del M_old

    # 循环结束
    print(f"Client {client.client_id} memory updated for Task {task_id}.")
