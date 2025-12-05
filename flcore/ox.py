import torch
from torch_geometric.nn import GCNConv

conv = GCNConv(10, 20)
# print(hasattr(conv, 'weight'))        # 可能 True，但已弃用
# print(conv.weight)                    # 可能 None 或 Parameter
print(hasattr(conv, 'lin'))           # ✅ True
print(conv.lin)                       # Linear(10 -> 20)