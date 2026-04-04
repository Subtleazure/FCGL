import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from flcore.base import BaseClient, BaseServer
from model import load_model
from torch_geometric.data import Data
from utils import *
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
import warnings

# 忽略 sklearn 聚类的一些常规警告
warnings.filterwarnings("ignore")


class OursClient(BaseClient):

    def __init__(self, args, client_id, data, message_pool, device):
        super(OursClient, self).__init__(
            args, client_id, data, message_pool, device)
        self.local_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim,
                                      output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.optim = Adam(self.local_model.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cache = {}

        # Local Mask Memory
        self.accumulated_mask = None  # Tensor of shape[K+1, D, D]
        self.mask_threshold_quantile = 0.3  # Keep top 30% importance
        self.temp = 0.05  # Temperature for soft sigmoid

        # 提取原型的数量参数
        self.num_prototypes_per_class = 5

    def get_task_num_samples(self, task_id):
        task = self.data["task"][task_id]
        task_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
        return task_mask.sum()

    def execute(self, task_id, round_id):
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        task_data = self.task_data(task_id, whole_data, task)

        # Initialize local model from global model
        self._update_local_from_global()

        # Get Pseudo Data (Prototypes) and Old Model from Server
        pseudo_data = self.message_pool["server"].get("pseudo_data", None)
        old_global_model = self.message_pool["server"].get(
            "old_global_model", None)

        self.local_model.train()

        for epoch_i in range(self.args.num_epochs):
            self.optim.zero_grad()

            # Forward pass on current task data
            # --- 前向传播 ---
            if self.args.model == "jacobi":
                logits, embedding, _, _ = self.local_model.forward(task_data)
            else:
                logits, embedding, _ = self.local_model.forward(task_data)

            if round_id % 10 == 0 and self.args.debug:  # 每 10 轮打印一次
                preds = logits[task["train_mask"]].argmax(dim=1)
                pred_counts = torch.bincount(
                    preds, minlength=self.args.output_dim)
                print(
                    f"[DEBUG] Client {self.client_id} Epoch {epoch_i} 预测的各类别数量: {pred_counts.tolist()}")

                emb_max = embedding.abs().max().item()
                logit_max = logits.abs().max().item()
                print(
                    f"[DEBUG] Client {self.client_id} Emb Max: {emb_max:.2f}, Logit Max: {logit_max:.2f}")

            # --- 1. 当前任务的学习 (Local Plasticity) ---
            train_mask = task["train_mask"]
            loss_ce = self.loss_fn(
                logits[train_mask], whole_data.y[train_mask])
            loss = loss_ce

            # -------------------------------------------------------------
            # --- 2. 弹性谱特征对齐 (Elastic Directional Stability) ---
            # -------------------------------------------------------------
            if old_global_model is not None:
                with torch.no_grad():
                    _, z_old, _, _ = old_global_model(task_data)

                loss_feat_stability = 1.0 - F.cosine_similarity(
                    embedding[train_mask],
                    z_old[train_mask],
                    dim=1,
                    eps=1e-8
                ).mean()

                loss += self.args.lam_feat * loss_feat_stability  # 5.0

            # -------------------------------------------------------------
            # --- 3. 动态过滤的全局回放 (Dynamic Filtered Replay) ---
            # -------------------------------------------------------------
            if task_id > 0 and pseudo_data is not None and self.args.gene:
                p_features_all, p_labels_all, p_golden_logits_all = pseudo_data

                # 【修改 2】：提取当前任务正在学习的真实类别
                current_learning_classes = torch.unique(
                    whole_data.y[train_mask]).cpu()

                # 构建掩码：滤除掉那些当前正在用真实数据训练的类别！只复习“真正见不到的旧类”
                p_labels_cpu = p_labels_all.cpu()
                valid_mask = ~torch.isin(
                    p_labels_cpu, current_learning_classes)

                # 如果过滤后还有旧类需要复习
                if valid_mask.sum() > 0:
                    p_features = p_features_all[valid_mask].to(self.device)
                    p_labels = p_labels_all[valid_mask].to(self.device)
                    p_golden_logits = p_golden_logits_all[valid_mask].to(
                        self.device)

                    student_logits_proto = self.local_model.classifier(
                        p_features)

                    # 【修改 3】：完全相信 Golden Logits，弱化甚至关闭 Hard CE
                    # Hard CE 会因为长程特征漂移导致冲突，而 Soft KL 包含的暗知识才是防遗忘的神器
                    loss_replay_hard = self.loss_fn(
                        student_logits_proto, p_labels)

                    T = self.args.T
                    loss_replay_soft = F.kl_div(
                        F.log_softmax(student_logits_proto / T, dim=1),
                        F.softmax(p_golden_logits / T, dim=1),
                        reduction='batchmean'
                    ) * (T * T)

                    # 参数建议：将拉锯战交给软标签。Hard CE降为 0.5 或者 0.0
                    loss += self.args.lam_re_hard * loss_replay_hard + \
                        self.args.lam_re_soft * loss_replay_soft

            # --- 反向传播 ---
            if epoch_i == 0 or epoch_i == self.args.num_epochs - 1 and self.args.debug:
                ce_val = loss_ce.item()

                # 假设你定义了这两个 loss (如果没有这部分，改成你实际的名字)
                feat_val = loss_feat_stability.item() if 'loss_feat_stability' in locals() else 0
                replay_hard_val = loss_replay_hard.item() if 'loss_replay_hard' in locals() else 0
                replay_soft_val = loss_replay_soft.item() if 'loss_replay_soft' in locals() else 0

                print(
                    f"[DEBUG] Client {self.client_id} Task {task_id} | CE: {ce_val:.4f} | Feat: {feat_val:.4f} | Replay Hard: {replay_hard_val:.4f} | Replay Soft: {replay_soft_val:.4f}")
            loss.backward()

            # --- NaN 探针 ---
            # 检查 loss 是否已经是 NaN
            if torch.isnan(loss).any():
                print(
                    f"[DEBUG] Client {self.client_id} Task {task_id}: LOSS 出现 NaN! 赶紧终止程序。")
                import sys
                sys.exit(1)

            # 检查梯度是否爆炸或变成 NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.local_model.parameters(), max_norm=100.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(
                    f"[DEBUG] Client {self.client_id} Task {task_id}: 梯度爆炸了! Grad Norm: {grad_norm}")

            # --- C. Apply Spectral Mask (Gradient Modification) ---
            if self.accumulated_mask is not None and self.args.model == "jacobi" and self.args.para:
                mask = self.accumulated_mask.to(self.device)
                self.local_model.W_weight.grad.data.mul_(1.0 - mask)

            self.optim.step()

        self.local_model.eval()

    def task_done(self, task_id):
        if self.args.model != "jacobi":
            return

        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        task_data = self.task_data(task_id, whole_data, task)

        if self.args.para:
            self.local_model.zero_grad()
            logits, _, _, alpha = self.local_model.forward(task_data)
            loss = self.loss_fn(logits[task["train_mask"]],
                                whole_data.y[task["train_mask"]])
            loss.backward()
            with torch.no_grad():
                grads = self.local_model.W_weight.grad.abs()
                avg_alpha = alpha.mean(dim=0).view(-1, 1, 1)
                score = grads * avg_alpha

                for k in range(score.shape[0]):
                    s_k = score[k]
                    min_v = s_k.min()
                    max_v = s_k.max()
                    norm_s = (s_k - min_v) / (max_v - min_v + 1e-8)
                    kth_val = torch.quantile(
                        norm_s.flatten(), 1.0 - self.mask_threshold_quantile)
                    mask_k = torch.sigmoid((norm_s - kth_val) / self.temp)
                    score[k] = mask_k

                current_mask = score
                if self.accumulated_mask is None:
                    self.accumulated_mask = current_mask.detach().clone()
                else:
                    self.accumulated_mask = torch.max(
                        self.accumulated_mask, current_mask.detach().clone())

        # --- 2. Spectral Prototype Extraction ---
        self.local_model.eval()
        if self.args.gene:
            with torch.no_grad():
                _, z_tilde, _, _ = self.local_model.forward(task_data)
                labels = whole_data.y
                mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
                z_curr = z_tilde[mask]
                y_curr = labels[mask]

                stats = {}
                unique_classes = torch.unique(y_curr)
                for c in unique_classes:
                    c_idx = (y_curr == c)
                    features_c = z_curr[c_idx].cpu().numpy()  # [N_c, D]

                    # 提取该类的代表性原型
                    num_samples = features_c.shape[0]
                    num_clusters = min(
                        self.num_prototypes_per_class, num_samples)

                    if num_clusters < 2:
                        prototypes = torch.tensor(features_c)
                    else:
                        try:
                            # 优先尝试谱聚类 (捕获流形结构)
                            sc = SpectralClustering(
                                n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
                            cluster_labels = sc.fit_predict(features_c)
                        except:
                            # 如果近邻图不连通等异常，退化为 KMeans
                            kmeans = KMeans(
                                n_clusters=num_clusters, random_state=42)
                            cluster_labels = kmeans.fit_predict(features_c)

                        # 计算每个聚类簇的质心作为 Prototype
                        proto_list = []
                        for i in range(num_clusters):
                            cluster_points = features_c[cluster_labels == i]
                            if cluster_points.shape[0] > 0:
                                proto_list.append(torch.tensor(
                                    cluster_points.mean(axis=0)))
                        prototypes = torch.stack(proto_list)

                    # # [LDP Privacy Guarantee] 注入轻微高斯噪声，防止隐私推断
                    # noise_scale = 0.01
                    # noise = torch.randn_like(prototypes) * noise_scale
                    # prototypes = prototypes + noise

                    stats[c.item()] = prototypes

            self.message_pool[f"client_{self.client_id}_stats"] = stats

    def _update_local_from_global(self):
        with torch.no_grad():
            for local_param, global_param in zip(self.local_model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

    def send_message(self, task_id):
        mask_to_send = self.accumulated_mask.cpu(
        ) if self.accumulated_mask is not None else None
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.get_task_num_samples(task_id),
            "weight": list(self.local_model.parameters()),
            "mask": mask_to_send
        }

    def evaluate(self, task_id, use_global=False, mask="test_mask"):
        if use_global:
            local_param_copy = copy.deepcopy(
                list(self.local_model.parameters()))
            with torch.no_grad():
                for (local_param, global_param) in zip(self.local_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)

        self.local_model.eval()
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]

        if task_id not in self.cache:
            task_data = self.task_data(task_id, whole_data, task)
            self.cache[task_id] = task_data

        task_data = self.cache[task_id]

        logits, embedding, _, _ = self.local_model.forward(task_data)
        acc_task_test = accuracy(logits[task[mask]], whole_data.y[task[mask]])

        if use_global:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.local_model.parameters(), local_param_copy):
                    local_param.data.copy_(global_param)

        return acc_task_test

    def task_data(self, task_id, whole_data, task):
        handled = task["train_mask"] | task["val_mask"] | task["test_mask"]
        masked_edge_index_filename = os.path.join(
            self.args.task_dir, f"client_{self.client_id}_task_{task_id}.pt")
        if not os.path.exists(masked_edge_index_filename):
            masked_edge_index = edge_masking(
                whole_data.edge_index, handled=handled, device=self.device)
            torch.save(masked_edge_index, masked_edge_index_filename)
        else:
            masked_edge_index = torch.load(
                masked_edge_index_filename, map_location=self.device)

        task_data = Data(
            x=whole_data.x, edge_index=masked_edge_index, y=whole_data.y)
        return task_data


class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool, device)
        self.global_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim,
                                       output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.old_global_model = None

        # --- Memory Bank Component ---
        self.global_memory_bank = {}  # {class_id: tensor(Num_Protos, D)}
        self.pseudo_data = None  # (features_tensor, labels_tensor)
        self.max_protos_per_class = 100  # 限制 Memory Bank 大小，防止爆炸

    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum(
                [self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in range(self.args.num_clients)])

            for i, (name, global_param) in enumerate(self.global_model.named_parameters()):
                global_param.data.zero_()
                is_spectral_weight = ("W_weight" in name) and (
                    "jacobi" in self.args.model)
                sum_mask = 0.0 + 1e-8

                for client_id in range(self.args.num_clients):
                    client_data = self.message_pool[f"client_{client_id}"]
                    local_param = client_data["weight"][i].to(self.device)
                    sample_weight = client_data["num_samples"] / \
                        num_tot_samples

                    if is_spectral_weight and self.args.para:
                        client_mask = client_data.get("mask", None)
                        if client_mask is not None:
                            mask = client_mask.to(self.device)
                            global_param.data += mask * local_param
                            sum_mask += mask
                        else:
                            global_param.data += sample_weight * local_param
                    else:
                        global_param.data += sample_weight * local_param

                if is_spectral_weight and self.args.para:
                    has_mask = any([self.message_pool[f"client_{cid}"].get(
                        "mask") is not None for cid in range(self.args.num_clients)])
                    if has_mask:
                        global_param.data.div_(sum_mask)

        self.global_model.eval()

    def task_done(self, task_id):
        """
        Server lifecycle at end of task:
        1. Save Old Model (Teacher for Feature Alignment).
        2. Collect new prototypes from clients.
        3. Compress Prototypes (Spectral Clustering).
        4. [核心创新] Compute Golden Logits at PEAK performance!
        5. Prepare Data (Features, Labels, Golden Logits) for next round.
        """
        if self.args.gene:
            # 1. 保存旧模型，仅用于客户端计算特征层面的 MSE Loss (Feature Alignment)
            self.old_global_model = copy.deepcopy(self.global_model)
            for param in self.old_global_model.parameters():
                param.requires_grad = False
            self.old_global_model.eval()

            # 2. 收集当前 Task 各个 Client 上传的最新原型
            current_task_protos = {}
            for client_id in range(self.args.num_clients):
                c_stats = self.message_pool.get(
                    f"client_{client_id}_stats", {})
                for c, protos in c_stats.items():
                    if c not in current_task_protos:
                        current_task_protos[c] = protos.cpu()
                    else:
                        current_task_protos[c] = torch.cat(
                            [current_task_protos[c], protos.cpu()], dim=0)

            # 3. 谱压缩与黄金 Logits 缓存
            self.global_model.eval()
            for c, protos in current_task_protos.items():
                num_protos = protos.shape[0]

                # --- A. 谱聚类压缩 (Scalability) ---
                if num_protos > self.max_protos_per_class:
                    protos_np = protos.numpy()
                    try:
                        # 捕获流形拓扑
                        sc = SpectralClustering(
                            n_clusters=self.max_protos_per_class, affinity='nearest_neighbors', random_state=42)
                        cluster_labels = sc.fit_predict(protos_np)
                    except:
                        # 降级方案
                        kmeans = KMeans(
                            n_clusters=self.max_protos_per_class, random_state=42)
                        cluster_labels = kmeans.fit_predict(protos_np)

                    new_protos = []
                    for i in range(self.max_protos_per_class):
                        cluster_points = protos[cluster_labels == i]
                        if cluster_points.shape[0] > 0:
                            new_protos.append(cluster_points.mean(dim=0))

                    compressed_protos = torch.stack(new_protos)
                else:
                    compressed_protos = protos

                # --- B. 提取并永久缓存巅峰状态的暗知识 (Golden Logits) ---
                compressed_protos = compressed_protos.to(self.device)
                with torch.no_grad():
                    # 用刚训练完当前 Task 的、记忆最清晰的全局分类器进行前向传播
                    golden_logits = self.global_model.classifier(
                        compressed_protos).cpu()

                # 特征转回 CPU 存储，节省显存
                compressed_protos = compressed_protos.cpu()

                # --- C. 存入全局记忆库 (Memory Bank Update) ---
                if c not in self.global_memory_bank:
                    self.global_memory_bank[c] = {
                        "features": compressed_protos,
                        "logits": golden_logits
                    }
                else:
                    # 如果该类在之前的 Task 出现过，我们合并后保留最新的以防爆炸
                    old_features = self.global_memory_bank[c]["features"]
                    old_logits = self.global_memory_bank[c]["logits"]

                    combined_features = torch.cat(
                        [old_features, compressed_protos], dim=0)
                    combined_logits = torch.cat(
                        [old_logits, golden_logits], dim=0)

                    if combined_features.shape[0] > self.max_protos_per_class:
                        # 【关键修复】：不能只取最新的！必须随机打乱抽样，保证新旧知识共存
                        perm = torch.randperm(combined_features.shape[0])
                        idx = perm[:self.max_protos_per_class]

                        combined_features = combined_features[idx]
                        combined_logits = combined_logits[idx]

                    self.global_memory_bank[c] = {
                        "features": combined_features,
                        "logits": combined_logits
                    }

            # 4. 准备下发给 Client 的 Pseudo Data
            self.pseudo_data = self.prepare_data()

    def prepare_data(self):
        """Flatten the dictionary into (Features, Labels, Logits) for the clients"""
        features_list = []
        labels_list = []
        logits_list = []

        if len(self.global_memory_bank) == 0:
            return None

        # 将字典解包成三个并列的 Tensor
        for c, data_dict in self.global_memory_bank.items():
            protos = data_dict["features"]
            g_logits = data_dict["logits"]
            num_protos = protos.shape[0]

            c_tensor = torch.full((num_protos,), c, dtype=torch.long)

            features_list.append(protos)
            labels_list.append(c_tensor)
            logits_list.append(g_logits)

        # 拼接并返回，供客户端在蒸馏时解包使用
        return torch.cat(features_list), torch.cat(labels_list), torch.cat(logits_list)

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.global_model.parameters()),
            "old_global_model": self.old_global_model,
            "pseudo_data": self.pseudo_data
        }
