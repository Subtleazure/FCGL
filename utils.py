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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from curves import *


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


class SpectralGenerator(nn.Module):
    """
    全局谱特征生成器
    输入: Noise + Class Label (Embedding)
    输出: Z_tilde (MLP+Jacobi后的聚合特征)
    """

    def __init__(self, num_classes, hidden_dim, noise_dim=64):
        super(SpectralGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, hidden_dim)  # 输出特征维度
        )

    def forward(self, noise, labels):
        # noise: [B, noise_dim]
        # labels: [B]
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        return self.net(x)


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
        if args.print:
            print(f"Task {task_id} starts.")
        FL_acc_matrix[task_id, :] = 0
        message_pool["task_id"] = task_id
        for client_id in range(args.num_clients):
            clients[client_id].task_start(task_id)
        server.task_start(task_id)
        accuracies = []
        for round_id in range(args.num_rounds_per_task):
            if args.print:
                print(f"Round {round_id} starts.")
            message_pool["round_id"] = round_id
            server.send_message()
            for client_id in range(args.num_clients):
                if args.print:
                    print(f"Client {client_id} is training...")
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
            clients[client_id].send_message(task_id)
        server.execute()
        server.task_done(task_id)
        print("-"*50)

    return aa, af


def plot_accuracy_curve(self, accuracies):
    """
    绘制原始准确率曲线和拟合图像，并保存为 HTML 文件和图片文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
    """

    # 绘制拟合图像
    if self.args.accuracy_curve_html_dir:
        plot_smoothed_accuracy_plotly(
            accuracies, self.args.accuracy_curve_html_dir, self.args.window_len)
    else:
        plot_smoothed_accuracy_plotly(accuracies, "accuracy_curve.html")

    if self.args.accuracy_curve_dir:
        plot_smoothed_accuracy_matplotlib(
            accuracies, self.args.accuracy_curve_dir, self.args.window_len)
    else:
        plot_smoothed_accuracy_matplotlib(accuracies, "accuracy_curve.png")


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


def _core_sgp_update(device, Z_data, M_old, lam_old,
                     args, importance_weight=1.0,
                     layer_name="Unknown"):
    """
    通用 SGP 更新核心逻辑
    Z_data: [N, D] 输入特征
    M_old, lam_old: 旧基底和重要性
    importance_weight: 外部传入的重要性权重 (如 alpha), 默认为 1.0
    """
    batch_size = args.batch_size
    epsilon = args.epsilon   # 注意：原代码用的 args.epsilon，这里统一用 threshold
    gamma = args.gamma

    num_nodes, feat_dim = Z_data.shape

    # 1. 计算全图协方差矩阵 (分块计算)
    Cov_total = torch.zeros((feat_dim, feat_dim), device=device)

    with torch.no_grad():
        for i in range(0, num_nodes, batch_size):
            batch_Z = Z_data[i: i + batch_size].to(device)
            Cov_total += torch.matmul(batch_Z.t(), batch_Z)

    total_energy = torch.trace(Cov_total)

    # 如果没有能量 (全0数据)，直接返回旧的
    if total_energy == 0:
        return (M_old, lam_old) if M_old is not None else (None, None)

    # 2. 准备旧基底
    if M_old is not None:
        M_old = M_old.to(device)
        lam_old = lam_old.to(device)

        # --- 计算代理奇异值 ---
        Cov_proj = torch.mm(M_old.t(), torch.mm(Cov_total, M_old))
        sigmas_proxy_sq = torch.diagonal(Cov_proj)
        sigmas_proxy = torch.sqrt(torch.abs(sigmas_proxy_sq))

        # --- 计算新基底 (Residual) ---
        MMT_C = torch.mm(torch.mm(M_old, M_old.t()), Cov_total)
        term_last = torch.mm(M_old, torch.mm(Cov_proj, M_old.t()))
        Cov_res = Cov_total - MMT_C - MMT_C.t() + term_last
        Cov_res = (Cov_res + Cov_res.t()) / 2  # 对称化

        eigvals_res, V_res = torch.linalg.eigh(Cov_res)

        # 排序
        idx = torch.argsort(eigvals_res, descending=True)
        eigvals_res = eigvals_res[idx]
        V_res = V_res[:, idx]

        # 截断
        energy_old = torch.sum(torch.square(sigmas_proxy))
        current_energy = energy_old
        target_energy = epsilon * total_energy

        num_new = 0
        for i in range(len(eigvals_res)):
            if current_energy >= target_energy:
                break
            if eigvals_res[i] > 0:
                current_energy += eigvals_res[i]
                num_new += 1

        if num_new > 0:
            M_new = V_res[:, :num_new]
            sigmas_new = torch.sqrt(eigvals_res[:num_new])
        else:
            M_new = torch.empty((feat_dim, 0), device=device)
            sigmas_new = torch.tensor([], device=device)

        # --- 联合归一化 & 更新 ---
        all_sigmas = torch.cat([sigmas_proxy, sigmas_new])

        if len(all_sigmas) > 0:
            max_sigma_total = torch.max(all_sigmas)
            # SGP 公式
            lam_all_raw = ((gamma + 1) * all_sigmas) / \
                (gamma * all_sigmas + max_sigma_total + 1e-8)

            # 乘以重要性权重 (对于 Spectral 层是 alpha, 其他层是 1.0)
            lam_all_weighted = lam_all_raw * importance_weight

            # 拆分
            r_old = len(sigmas_proxy)
            lam_proxy_weighted = lam_all_weighted[:r_old]
            lam_new_weighted = lam_all_weighted[r_old:]

            # 历史 Max-Pooling
            lam_old_updated = torch.max(lam_old, lam_proxy_weighted)

            M_final = torch.cat([M_old, M_new], dim=1)
            lam_final = torch.cat([lam_old_updated, lam_new_weighted])
        else:
            M_final, lam_final = M_old, lam_old

    else:
        # --- 第一次任务 ---
        eigvals, V = torch.linalg.eigh(Cov_total)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        V = V[:, idx]

        current_energy = 0
        target_energy = epsilon * total_energy
        num_new = 0
        for i in range(len(eigvals)):
            if current_energy >= target_energy:
                break
            current_energy += eigvals[i]
            num_new += 1

        M_final = V[:, :num_new]
        sigmas = torch.sqrt(eigvals[:num_new])

        if num_new > 0:
            max_sig = sigmas[0]
            lam_raw = ((gamma + 1) * sigmas) / \
                (gamma * sigmas + max_sig + 1e-8)
            lam_final = lam_raw * importance_weight
        else:
            lam_final = torch.tensor([], device=device)

    # 清理显存并返回 CPU Tensor
    # lam_final都是1
    # lam_final = torch.ones_like(lam_final)
    del Cov_total
    return M_final.cpu(), lam_final.cpu()

# def _core_sgp_update(device, Z_data, M_old, lam_old,
#                      args, importance_weight=1.0,
#                      layer_name="Unknown"):
#     batch_size = args.batch_size
#     epsilon = args.epsilon
#     gamma = args.gamma

#     num_nodes, feat_dim = Z_data.shape

#     # 1. 计算全图协方差矩阵
#     Cov_total = torch.zeros((feat_dim, feat_dim), device=device)
#     with torch.no_grad():
#         for i in range(0, num_nodes, batch_size):
#             batch_Z = Z_data[i: i + batch_size].to(device)
#             Cov_total += torch.matmul(batch_Z.t(), batch_Z)

#     total_energy = torch.trace(Cov_total)
#     if total_energy == 0:
#         return (M_old, lam_old) if M_old is not None else (None, None)

#     # 2. 准备旧基底
#     if M_old is not None:
#         M_old = M_old.to(device)
#         lam_old = lam_old.to(device)

#         # --- 计算代理奇异值 ---
#         Cov_proj = torch.mm(M_old.t(), torch.mm(Cov_total, M_old))
#         sigmas_proxy_sq = torch.diagonal(Cov_proj)
#         sigmas_proxy = torch.sqrt(torch.abs(sigmas_proxy_sq))

#         # --- 【修改A】: 单独计算旧基底的当前重要性 (不与新基底混合) ---
#         # 使用旧基底内部的最大值进行归一化，保持其绝对地位
#         max_sigma_proxy = sigmas_proxy.max() if len(sigmas_proxy) > 0 else 1.0

#         # 计算当前任务看来，旧基底的重要性
#         lam_proxy_raw = ((gamma + 1) * sigmas_proxy) / \
#                         (gamma * sigmas_proxy + max_sigma_proxy + 1e-8)

#         # 乘以当前任务的关注度 (alpha)
#         # 注意: 即使当前 alpha 很小，max操作也会由 lam_old 兜底
#         lam_proxy_weighted = lam_proxy_raw * importance_weight

#         # --- 历史更新 (Max-Pooling) ---
#         # 这是防遗忘的根本：只要历史上重要过一次，就永远重要
#         lam_old_updated = torch.max(lam_old, lam_proxy_weighted)

#         # --- 计算新基底 (Residual) ---
#         MMT_C = torch.mm(torch.mm(M_old, M_old.t()), Cov_total)
#         term_last = torch.mm(M_old, torch.mm(Cov_proj, M_old.t()))
#         Cov_res = Cov_total - MMT_C - MMT_C.t() + term_last
#         Cov_res = (Cov_res + Cov_res.t()) / 2

#         eigvals_res, V_res = torch.linalg.eigh(Cov_res)

#         # 排序 & 截断
#         idx = torch.argsort(eigvals_res, descending=True)
#         eigvals_res = eigvals_res[idx]
#         V_res = V_res[:, idx]

#         # 能量截断：这里依然需要考虑整体能量，以决定加多少新基底
#         # 使用 proxy 能量而不是 lam 来计算已保留能量
#         energy_old = torch.sum(torch.square(sigmas_proxy))
#         current_energy = energy_old
#         target_energy = epsilon * total_energy

#         num_new = 0
#         for i in range(len(eigvals_res)):
#             if current_energy >= target_energy:
#                 break
#             if eigvals_res[i] > 0:
#                 current_energy += eigvals_res[i]
#                 num_new += 1

#         if num_new > 0:
#             M_new = V_res[:, :num_new]
#             sigmas_new = torch.sqrt(eigvals_res[:num_new])

#             # --- 【修改B】: 单独计算新基底的重要性 ---
#             max_sigma_new = sigmas_new[0]
#             lam_new_raw = ((gamma + 1) * sigmas_new) / \
#                           (gamma * sigmas_new + max_sigma_new + 1e-8)
#             lam_new_weighted = lam_new_raw * importance_weight

#             # 拼接
#             M_final = torch.cat([M_old, M_new], dim=1)
#             lam_final = torch.cat([lam_old_updated, lam_new_weighted])
#         else:
#             M_final = M_old
#             lam_final = lam_old_updated

#     else:
#         # --- 第一次任务 ---
#         eigvals, V = torch.linalg.eigh(Cov_total)
#         idx = torch.argsort(eigvals, descending=True)
#         eigvals = eigvals[idx]
#         V = V[:, idx]

#         current_energy = 0
#         target_energy = epsilon * total_energy
#         num_new = 0
#         for i in range(len(eigvals)):
#             if current_energy >= target_energy:
#                 break
#             current_energy += eigvals[i]
#             num_new += 1

#         M_final = V[:, :num_new]
#         sigmas = torch.sqrt(eigvals[:num_new])

#         if num_new > 0:
#             max_sig = sigmas[0]
#             lam_raw = ((gamma + 1) * sigmas) / \
#                 (gamma * sigmas + max_sig + 1e-8)
#             lam_final = lam_raw * importance_weight
#         else:
#             lam_final = torch.tensor([], device=device)

#     del Cov_total
#     return M_final.cpu(), lam_final.cpu()


def update_memory_with_sgp(args, client, task_id, Zs_stack, alpha, whole_x_data=None, Z_tilde_data=None):
    """
    升级版: 同时更新 Spectral Layers (M_k), MLP (M_mlp), Classifier (M_cls)
    新增参数:
    - whole_x_data: 原始输入特征 X (用于 MLP) [N, In_Dim]
    - Z_tilde_data: 融合后的特征 Z_tilde (用于 Classifier) [N, Hid_Dim]
    """
    device = client.device

    # -------------------------------------------------------------
    # 1. 更新 Spectral Layers (核心部分)
    # -------------------------------------------------------------
    K = Zs_stack.shape[0] - 1

    # 初始化
    # if not hasattr(client, 'M_k') or client.M_k is None:
    #     client.M_k = {k: None for k in range(K + 1)}
    #     client.lam_k = {k: None for k in range(K + 1)}

    for k in range(K + 1):
        # 计算该阶的平均 Attention 权重
        avg_alpha = alpha[:, k].mean().item()

        # 调用核心更新函数
        M_updated, lam_updated = _core_sgp_update(
            device, Zs_stack[k],
            client.M[k], client.lam[k],
            args, importance_weight=avg_alpha,
            layer_name=f"Spectral_{k}"
        )

        client.M[k] = M_updated
        client.lam[k] = lam_updated

    # -------------------------------------------------------------
    # 2. 更新 Classifier (强烈推荐，开销极小)
    # -------------------------------------------------------------
    if Z_tilde_data is not None:
        if not hasattr(client, 'M_cls'):
            client.M_cls = None
            client.lam_cls = None

        # Classifier 对所有样本都生效，importance_weight = 1.0
        # 输入是 Z_tilde [N, Hid_Dim]
        # tmp = args.gamma
        # temp = args.epsilon
        # args.gamma = 1000
        # args.epsilon = 0.99
        M_updated_, lam_updated_ = _core_sgp_update(
            device, Z_tilde_data,
            client.M_cls, client.lam_cls,
            args, importance_weight=1.0,
            layer_name="Classifier"
        )
        # args.gamma = tmp
        # args.epsilon = temp

        client.M_cls = M_updated_
        client.lam_cls = lam_updated_

    # -------------------------------------------------------------
    # 3. 更新 MLP (可选，视输入维度而定)
    # -------------------------------------------------------------
    # 策略: 如果是 Task 0，或者显存足够，则更新。
    # 否则可以选择跳过 (Freezing MLP策略)

    update_mlp = True
    # 如果维度过大，可以只在 Task 0 更新
    # if whole_x_data.shape[1] > 2000 and task_id > 0:
    #     update_mlp = False
    # print("Skipping MLP SGP update due to high dimensionality.")

    if whole_x_data is not None and update_mlp:
        if not hasattr(client, 'M_mlp'):
            client.M_mlp = None
            client.lam_mlp = None

        # MLP 对所有样本生效，importance_weight = 1.0
        # 输入是 Raw X [N, In_Dim]
        _M_updated, _lam_updated = _core_sgp_update(
            device, whole_x_data,
            client.M_mlp, client.lam_mlp,
            args, importance_weight=1.0,
            layer_name="MLP"
        )

        client.M_mlp = _M_updated
        client.lam_mlp = _lam_updated

    print(f"Client {client.client_id} full memory updated for Task {task_id}.")
