import pynvml
from config import args
import torch
import copy
from data import load_fcgl_dataset
from utils import load_clients_server, start, seed_everything
from random import randint
import torch_geometric

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr
])

# 如果后续还提示拦截了其他 PyG 对象（例如 Data），可以像下面这样一并加入：
# from torch_geometric.data import Data
# torch.serialization.add_safe_globals([
#     torch_geometric.data.data.DataEdgeAttr,
#     Data
# ])

first_initialized_copy = {"flag": False, "clients": None, "server": None}


def run_experiment():
    # 加载 clients, server, message_pool
    clients, server, message_pool = load_clients_server(
        args, fcgl_dataset, device)

    if first_initialized_copy["flag"]:
        # 设置参数
        for client_id in range(args.num_clients):
            with torch.no_grad():
                for (local_param_old, initialized_param) in zip(clients[client_id].local_model.parameters(), first_initialized_copy["clients"][client_id].local_model.parameters()):
                    local_param_old.data.copy_(initialized_param)

        with torch.no_grad():
            for (global_param_old, initialized_param) in zip(server.global_model.parameters(), first_initialized_copy["server"].global_model.parameters()):
                global_param_old.data.copy_(initialized_param)

    else:
        first_initialized_copy["flag"] = True
        first_initialized_copy["clients"] = copy.deepcopy(clients)
        first_initialized_copy["server"] = copy.deepcopy(server)

    return start(args, fcgl_dataset, clients, server, message_pool, device)


def get_best_gpu():
    # 初始化 NVML
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    best_gpu_id = 0
    max_free_memory = 0

    print("GPU 状态扫描:")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # 转换为 MB 以便观察
        free_mb = info.free / 1024**2
        total_mb = info.total / 1024**2
        print(f"GPU {i}: 剩余显存 {free_mb:.0f}MB / 总显存 {total_mb:.0f}MB")

        if free_mb > max_free_memory:
            max_free_memory = free_mb
            best_gpu_id = i

    pynvml.nvmlShutdown()
    return best_gpu_id


if __name__ == "__main__":
    # args.gpuid = randint(0, torch.cuda.device_count() - 1)  # 随机选择一个 GPU
    args.gpuid = get_best_gpu()  # 自动选择一个空闲的 GPU
    print(f"Using GPU {args.gpuid}")
    args.disable_cuda = False

    seed_everything(args.seed)
    if not args.disable_cuda:
        device = torch.device(f"cuda:{args.gpuid}")
    else:
        device = torch.device(f"cpu")

    fcgl_dataset, input_dim, output_dim, task_dir = load_fcgl_dataset(
        root=args.root,
        dataset=args.dataset,
        num_clients=args.num_clients,
        classes_per_task=args.num_classes_per_task,
        shuffle_task=args.shuffle_task
    )
    args.input_dim = input_dim  # feature dimension
    args.output_dim = output_dim  # all task classes
    args.task_dir = task_dir

    # 运行实验
    run_experiment()

    print("K:", args.K)
    print("rounds:", args.num_rounds_per_task)
    print("para:", args.para)
    print("gene:", args.gene)
    print("attention:", args.alpha_at)
    print("lam_feat:", args.lam_feat)
    print("lam_re_hard:", args.lam_re_hard)
    print("T:", args.T)
    print("lam_re_soft:", args.lam_re_soft)
    print("shuffle_task:", args.shuffle_task)
    print("num_rounds_per_task:", args.num_rounds_per_task)
