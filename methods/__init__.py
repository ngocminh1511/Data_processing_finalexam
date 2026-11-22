"""Danh sách các phương pháp chọn frame.

Thêm phương pháp mới:
  1. Copy thư mục methods/_template thành methods/<ten_phuong_phap>
  2. Cài đặt hàm select()
  3. Thêm một dòng vào REGISTRY bên dưới
"""
import importlib

# tên dùng trong --method  ->  (module, tên class)
REGISTRY = {
    "uniform": ("methods.uniform", "UniformSelector"),
    "aks": ("methods.aks", "AKSSelector"),
    "ktv": ("methods.ktv", "KTVSelector"),
    "mdp3": ("methods.mdp3", "MDP3Selector"),
    # "mopr":   ("methods.moprune", "MoPruneSelector"),
}


def get_method(name: str, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Không có phương pháp '{name}'. Hiện có: {list(REGISTRY)}")
    module_name, cls_name = REGISTRY[name]
    # import lười: chạy 'uniform' không cần cài torch/transformers
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
