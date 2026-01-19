from config import args
import torch
import copy
from data import load_fcgl_dataset
from utils import load_clients_server, start, seed_everything

# 手动设置超参数
args.beta = 0.05  # 可以设置为 0.01 到 0.1 之间的值
args.num_epoch_g = 5  # 可以设置为 3, 5 或 10

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

    return start(args, fcgl_dataset, clients, server, message_pool, device)[0]


if __name__ == "__main__":
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
    result = run_experiment()

    print('Experiment Result:')
    print('  Global AA: {}'.format(result))
    print('    beta: {}'.format(args.beta))
    print('    decay: {}'.format(args.decay))
    print('    num_epoch_g: {}'.format(args.num_epoch_g))
