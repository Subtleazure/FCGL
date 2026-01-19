import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from flcore.base import BaseClient, BaseServer
from model import load_model
from torch_geometric.data import Data
from openfgl.utils.basic_utils import idx_to_mask_tensor
from utils import *


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


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
        if args.model == "jacobi":
            self.M = [None] * (args.K + 1)
            self.lam = [None] * (args.K + 1)

    def get_task_num_samples(self, task_id):
        task = self.data["task"][task_id]
        task_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
        return task_mask.sum()

    def execute(self, task_id):
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        task_data = self.task_data(task_id, whole_data, task)

        # Initialize local model from global model
        self._update_local_from_global()

        self.local_model.train()

        for epoch_i in range(self.args.num_epochs):
            self.optim.zero_grad()

            # Forward pass on current task data
            if self.args.model == "jacobi":
                logits, embedding, _, _ = self.local_model.forward(task_data)
                # SVD 后添加基底, M 加 Z_k, 快收敛时 B 加

            else:
                logits, embedding, _ = self.local_model.forward(task_data)
            loss_ce = self.loss_fn(
                logits[task["train_mask"]], whole_data.y[task["train_mask"]])

            # TODO: 可以在这里加入 Global Loss (利用 M_g)
            # loss_global using M_g from server
            # loss_global = ...

            total_loss = loss_ce
            total_loss.backward()

            # =======================================================
            # 【SGP 梯度投影核心代码】
            # 仅在 task_id > 0 (即有历史记忆) 时执行
            # =======================================================
            if task_id > 0 and hasattr(self, 'M'):
                # 1. 约束 Jacobi 谱权重 (W_weight)
                # self.local_model.W_weight 形状: [K+1, hid, hid]
                # 我们需要对每一个 k 分别做切片投影
                if self.local_model.W_weight.grad is not None:
                    for k in range(self.local_model.K + 1):
                        # 获取第 k 阶的梯度 [hid, hid]
                        grad_k = self.local_model.W_weight.grad[k]

                        # 获取对应的历史基底和重要性
                        # 注意：需要确保 M_k 字典里有对应 k 的数据
                        if k < len(self.M) and self.M[k] is not None:
                            M = self.M[k]
                            Lam = self.lam[k]

                            # 执行左投影 (因为 forward 是 Z @ W)
                            projected_grad = self.project_tensor_grad(
                                grad_k, M, Lam, left_projection=True)

                            # 将修正后的梯度写回
                            self.local_model.W_weight.grad[k] = projected_grad

                # # 2. (可选但推荐) 约束 MLP 输入层
                # # 需要在 task 结束时也对原始 X 做 SVD 存入 self.M_mlp
                # if hasattr(self, 'M_mlp') and self.M_mlp is not None:
                #     grad_mlp = self.local_model.mlp.weight.grad
                #     if grad_mlp is not None:
                #         # Linear层，执行右投影
                #         self.local_model.mlp.weight.grad = self.project_tensor_grad(
                #             grad_mlp, self.M_mlp, self.lam_mlp, left_projection=False
                #         )

                # # 3. (可选但推荐) 约束 Classifier 输出层
                # # 需要在 task 结束时对 Z_tilde 做 SVD 存入 self.M_cls
                # if hasattr(self, 'M_cls') and self.M_cls is not None:
                #     grad_cls = self.local_model.classifier.weight.grad
                #     if grad_cls is not None:
                #         # Linear层，执行右投影
                #         self.local_model.classifier.weight.grad = project_tensor_grad(
                #             grad_cls, self.M_cls, self.lam_cls, left_projection=False
                #         )

            # =======================================================

            self.optim.step()

        self.local_model.eval()

    # 定义一个辅助函数来进行梯度投影计算
    # left_projection=True 用于 W_weight (Z @ W)
    # left_projection=False 用于 Linear层 (x @ W.T)
    def project_tensor_grad(self, grad, M, Lam, left_projection=True):
        if M is None or grad is None:
            return grad

        # 搬运到 GPU 进行计算
        M = M.to(self.device)
        Lam = Lam.to(self.device)

        # 计算投影 P = M * Lambda * M^T
        # 为了效率，我们不显式构建大的 P，而是利用结合律
        # G_new = G - M @ (Lam * (M^T @ G))  (Left)
        # G_new = G - (G @ M) @ (Lam * M^T)  (Right)

        if left_projection:
            # 针对 W_weight: [D, D]
            # 1. inner = M^T @ G  -> [r, D]
            inner = torch.mm(M.t(), grad)
            # 2. scaled = Lam * inner (广播乘法, Lam是向量 [r]) -> [r, D]
            # 注意：Lambda 对基底加权，相当于对 inner 的行加权
            scaled = Lam.view(-1, 1) * inner
            # 3. projection = M @ scaled -> [D, D]
            projection = torch.mm(M, scaled)
            return grad - projection
        else:
            # 针对 nn.Linear: [Out, In]
            # 1. inner = G @ M -> [Out, r]
            inner = torch.mm(grad, M)
            # 2. scaled = inner * Lam (广播乘法, Lam是向量 [r]) -> [Out, r]
            # Lambda 对基底加权，相当于对 inner 的列加权
            scaled = inner * Lam.view(1, -1)
            # 3. projection = scaled @ M^T -> [Out, In]
            projection = torch.mm(scaled, M.t())
            return grad - projection

    def _update_local_from_global(self):
        """Update local model parameters from global model"""
        with torch.no_grad():
            for local_param, global_param in zip(self.local_model.parameters(),
                                                 self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

    def send_message(self, task_id):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.get_task_num_samples(task_id),
            "weight": list(self.local_model.parameters())
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

    def encode_gradient(self, task, task_data):
        self.ge_model.train()
        proto_grad = []
        selected_nodes = task["train_mask"]
        ground_truth = task_data.y[selected_nodes]
        current_classes = torch.unique(ground_truth).tolist()
        for class_i in current_classes:
            class_i_prototype = torch.mean(
                task_data.x[selected_nodes][ground_truth == class_i], dim=0)
            num_class_i = (ground_truth == class_i).sum()
            print(f"[client {self.client_id} task {self.message_pool['task_id']} round {self.message_pool['round_id']}]\tclass: {class_i}\ttotal_nodes: {num_class_i}")

            outputs = self.ge_model.forward(class_i_prototype)
            loss_cls = nn.CrossEntropyLoss()(
                outputs, torch.tensor(class_i).long().to(self.device))
            dy_dx = torch.autograd.grad(loss_cls, self.ge_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append((original_dy_dx, num_class_i))
        return proto_grad


class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool, device)
        self.global_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim,
                                       output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.old_global_model = None  # Store previous global model for distillation

    def execute(self):
        # if self.old_global_model is not None:
        #     generator = GraphGenerator(
        #         nz=128,
        #         feat_dim=self.args.input_dim,
        #         num_classes=self.args.num_classes_per_task,
        #         max_nodes=self.args.max_nodes_per_graph)
        #     # 学生模型
        #     student = copy.deepcopy(self.old_global_model)
        #     student.apply(gcn_weight_init)
        #     synthesizer = GraphSynthesizer(
        #         synthesis_batch_size=self.args.synthesis_batch_size,
        #         teacher=self.old_global_model,
        #         student=student,
        #         generator=generator,
        #         args=self.args
        #     )

        #     # Step 2: 合成图数据（在服务器端）
        #     criterion = KLDiv(T=self.args.temperature)
        #     optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
        #                                 momentum=0.9)
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, self.args.synthesis_rounds, eta_min=2e-4)

        #     for it in range(self.args.synthesis_rounds):
        #         self.message_pool["synthesis_graphs"].append(
        #             synthesizer.synthesize())
        #         if it >= self.args.warmup:
        #             loader = self.get_all_syn_data()
        #             graph_batch_iter = DataIter(loader)
        #             kd_train(student, self.old_global_model,
        #                      graph_batch_iter, criterion,
        #                      optimizer, self.device, self.args.kd_epochs)
        #             # compute student model acc
        #         scheduler.step()

        with torch.no_grad():
            num_tot_samples = sum(
                [self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in range(self.args.num_clients)])
            for client_id in range(self.args.num_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / \
                    num_tot_samples
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.global_model.parameters()):
                    if client_id == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

        # self.old_global_model = copy.deepcopy(self.global_model)
        self.global_model.eval()

    def gradient2label(self, grad):
        pred = torch.argmin(
            grad[-1]).detach().reshape((1,)).requires_grad_(False)
        return pred.item()

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.global_model.parameters()),
            "old_global_model": self.old_global_model
        }

    def get_all_syn_data(self):
        # 直接从 message_pool 获取图数据列表
        graph_list = self.message_pool["synthesis_graphs"]

        if not graph_list:
            raise ValueError(
                "No synthetic graphs found in message_pool['synthesis_graphs']")

        # 创建图数据集
        syn_dataset = GraphDataset(graph_list)

        # 创建 DataLoader
        # 注意：对于图数据，通常不需要 shuffle（尤其是已 batch 化），但如果想打乱 batch 顺序可以保留
        loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=None,           # 已经是 batched graph，不需要再 batch
            shuffle=True,              # 打乱不同 batch 图之间的顺序
            num_workers=4,             # 图数据一般不支持多进程加载（尤其 DGL/PyG）
            pin_memory=True,          # 根据设备决定，通常图结构复杂，不建议开启
            collate_fn=lambda x: x[0]  # 直接返回第一个元素（因为 batch_size=None）
        )

        return loader
