import argparse
import sys

parser = argparse.ArgumentParser()


parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--disable_cuda", action="store_true", default=False)
parser.add_argument("--shuffle_task", action="store_true", default=False)
parser.add_argument("--isolate_mode", action="store_true", default=False)
parser.add_argument("--global_eval", action="store_true",
                    default=False)  # default: local model on local data


parser.add_argument(
    "--root", default="/data1/huangsuyuan/FCGL/your_dataset_root_path")


parser.add_argument("--repeat", type=int, default=10)

parser.add_argument("--num_clients", type=int, default=3)
parser.add_argument("--num_classes_per_task", type=int, default=2)
parser.add_argument("--dataset", type=str, default="CiteSeer")


parser.add_argument("--method", type=str, default="ours", choices=["ours"])


parser.add_argument("--client_frac", type=float, default=1.0)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--hid_dim", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--model", type=str, default="jacobi")


# config of Ours
parser.add_argument("--disable_kd", action="store_true", default=False)

# fixed
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--replay", type=str, default="CM",
                    choices=["random", "CM"])
parser.add_argument("--num_samples_per_class", type=int, default=1)
parser.add_argument("--lr_g", default=1e-2)
parser.add_argument("--LBFGS_init_lr", type=float, default=1.0)
parser.add_argument("--num_it_recon", type=int, default=300)


# explored
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--num_recon_nodes", type=int, default=1)
parser.add_argument("--decay", type=float, default=0.3)
parser.add_argument("--num_epoch_g", type=int, default=3)


##################################

parser.add_argument("--max_nodes_per_graph", type=int, default=500)
parser.add_argument("--synthesis_batch_size", type=int, default=8)
parser.add_argument("--synthesis_rounds", type=int, default=50)
parser.add_argument("--temperature", type=float, default=2)

parser.add_argument("--kd_epochs", type=int, default=200)
parser.add_argument("--warmup", type=int, default=20)

parser.add_argument("--epsilon", type=float, default=0.99)  # energy threshold
# importance scaling coefficient
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=10000)
# svd
# parser.add_argument("--M", type=list, default=[])  # 基底
parser.add_argument("--B", type=list, default=[])
parser.add_argument("--Omega", type=list, default=[])
parser.add_argument("--print", type=bool, default=False)

parser.add_argument("--para", type=bool, default=True)
parser.add_argument("--gene", type=bool, default=True)
parser.add_argument("--num_rounds_per_task", type=int, default=55)

parser.add_argument("--gpuid", type=int, default=0)

parser.add_argument("--K", type=int, default=4)  # 谱阶数

parser.add_argument("--alpha_at", type=bool, default=True)  # 是否使用注意力机制
args = parser.parse_args()
