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
    # --- 新增：日志重定向逻辑 ---
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")  # "a" 表示追加模式

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):  # 保证兼容性
            self.terminal.flush()
            self.log.flush()

    # 确定日志文件名
    log_filename = f"/data1/liujiaqi/FCGL_logs/log_{args.dataset}.txt"
    # 如果你想每次运行都覆盖旧日志，可以用 "w" 模式，或者在这里先删除旧文件
    sys.stdout = Logger(log_filename)
    # -----------------------
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

    num_tasks = len(fcgl_dataset[0]["task"])
    args.num_tasks = num_tasks

    # 1. 整理所有需要同步管理的根目录
    # 注意：确保 args 中这些路径没有以此结尾的斜杠，os.path.join 会自动处理
    if args.save:
        monitored_dirs = [
            args.accuracy_curve_html_dir,
            args.accuracy_curve_dir,
            args.aa_html_dir,
            args.aa_dir,
            args.af_html_dir,
            args.af_dir
        ]

        # 2. 获取本次实验的同步编号 (例如 "0", "1", "2"...)
        exp_id = get_synchronized_experiment_id(monitored_dirs)
        print(f"Current Experiment ID: {exp_id}")

        # 3. 创建对应的子文件夹，并获取路径映射
        # paths_map[args.accuracy_curve_dir] 将返回 ".../accuracy_curve_dir/0"
        paths_map = create_experiment_folders(monitored_dirs, exp_id)

    # start train
    print("-"*50)
    FL_acc_matrix = torch.zeros(size=(num_tasks, num_tasks)).to(device)
    af_list = []
    aa_list = []
    for _ in range(num_tasks - 1):
        aa_list.append([])
        af_list.append([])
    for task_id in range(num_tasks):
        if args.print:
            print(f"Task {task_id} starts.")
        message_pool["task_id"] = task_id
        for client_id in range(args.num_clients):
            clients[client_id].task_start(task_id)
        server.task_start(task_id)
        accuracies = []
        for _ in range(task_id + 1):
            # init accuracies list
            accuracies.append([])
        for round_id in range(args.num_rounds_per_task):
            FL_acc_matrix[task_id, :] = 0
            if args.print:
                print(f"Round {round_id} starts.")
            message_pool["round_id"] = round_id
            server.send_message()
            for client_id in range(args.num_clients):
                if args.print:
                    print(f"Client {client_id} is training...")
                clients[client_id].execute(task_id, round_id)
                clients[client_id].send_message(task_id)
            server.execute()
            if not args.save:
                continue
            for eval_task_id in range(0, task_id+1):
                total_nodes = 0
                FL_acc_matrix[task_id, eval_task_id] = 0
                for client_id in range(args.num_clients):
                    client_acc = clients[client_id].evaluate(
                        task_id=eval_task_id, use_global=args.global_eval)
                    num_nodes = clients[client_id].data["task"][eval_task_id]["test_mask"].sum(
                    )
                    FL_acc_matrix[task_id,
                                  eval_task_id] += client_acc * num_nodes
                    total_nodes += num_nodes
                FL_acc_matrix[task_id, eval_task_id] /= total_nodes
                accuracies[eval_task_id].append(
                    FL_acc_matrix[task_id, eval_task_id].item())
            if task_id > 0:
                aa = AA(FL_acc_matrix, T=task_id+1)
                af = AF(FL_acc_matrix, T=task_id+1)
                aa_list[task_id - 1].append(aa)
                af_list[task_id - 1].append(af)
        for i in range(task_id + 1):
            if not args.save:
                break
            save_html_dir = paths_map[args.accuracy_curve_html_dir]
            save_png_dir = paths_map[args.accuracy_curve_dir]
            html_path = os.path.join(
                save_html_dir, f"task_{i}_in_task_{task_id}_curve.html")
            png_path = os.path.join(
                save_png_dir, f"task_{i}_in_task_{task_id}_curve.png")
            plot_accuracy_curve(
                accuracies[i], html_path, png_path, window_len=args.window_len)

        # ↑ 训练  |  评估 ↓
        if not args.save:
            for eval_task_id in range(0, task_id+1):
                total_nodes = 0
                for client_id in range(args.num_clients):
                    client_acc = clients[client_id].evaluate(
                        task_id=eval_task_id, use_global=args.global_eval)
                    num_nodes = clients[client_id].data["task"][eval_task_id]["test_mask"].sum(
                    )
                    FL_acc_matrix[task_id,
                                  eval_task_id] += client_acc * num_nodes
                    total_nodes += num_nodes
                FL_acc_matrix[task_id, eval_task_id] /= total_nodes
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

    for t_id in range(1, num_tasks):
        if not args.save:
            break
        aa_html_path = os.path.join(
            paths_map[args.aa_html_dir], f"task_{t_id}_aa_curve.html")
        aa_png_path = os.path.join(
            paths_map[args.aa_dir], f"task_{t_id}_aa_curve.png")

        plot_accuracy_curve(
            aa_list[t_id - 1], aa_html_path, aa_png_path, window_len=args.window_len)

        af_html_path = os.path.join(
            paths_map[args.af_html_dir], f"task_{t_id}_af_curve.html")
        af_png_path = os.path.join(
            paths_map[args.af_dir], f"task_{t_id}_af_curve.png")

        plot_accuracy_curve(
            af_list[t_id - 1], af_html_path, af_png_path, window_len=args.window_len)

    if args.save:
        print(f"All plots saved to folder ID: {exp_id}")


def get_synchronized_experiment_id(base_dirs):
    """
    检查传入的所有 base_dirs 列表。
    返回一个在所有 base_dirs 中都未被占用的最小整数编号（字符串格式）。
    """
    exp_id = 0
    while True:
        # 检查当前的 exp_id 是否在任意一个 base_dir 中已存在
        conflict = False
        for base_dir in base_dirs:
            # 检查 base_dir/0, base_dir/1 ... 是否存在
            check_path = os.path.join(base_dir, str(exp_id))
            if os.path.exists(check_path):
                conflict = True
                break

        # 如果当前 exp_id 在所有目录中都不冲突，则选中它
        if not conflict:
            return str(exp_id)

        # 否则尝试下一个编号
        exp_id += 1


def create_experiment_folders(base_dirs, exp_id):
    """
    根据选定的 exp_id，在所有 base_dirs 下创建对应的文件夹。
    返回创建好的完整路径字典，方便调用。
    """
    created_paths = {}
    for base_dir in base_dirs:
        full_path = os.path.join(base_dir, exp_id)
        os.makedirs(full_path, exist_ok=True)
        created_paths[base_dir] = full_path
    return created_paths


def plot_accuracy_curve(accuracies, html_path, png_path, window_len):
    """
    绘制原始准确率曲线和拟合图像，并保存为 HTML 文件和图片文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
    """

    # 绘制拟合图像
    accuracies = [acc.item() if torch.is_tensor(
        acc) else acc for acc in accuracies]
    if html_path:
        plot_smoothed_accuracy_plotly(
            accuracies, html_path, window_len)
    else:
        plot_smoothed_accuracy_plotly(accuracies, "accuracy_curve.html")

    if png_path:
        plot_smoothed_accuracy_matplotlib(
            accuracies, png_path, window_len)
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
