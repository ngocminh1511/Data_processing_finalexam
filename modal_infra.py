"""Hạ tầng Modal dùng chung: Volume và Image.

Code trong thư mục methods/ và các file config.py, modal_infra.py được
upload lên container qua add_local_python_source.
"""
import modal

from config import HF_PATH, HF_VOL_NAME, VOL_NAME, VOL_PATH

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOL_NAME, create_if_missing=True)
VOLUMES = {VOL_PATH: vol, HF_PATH: hf_vol}

# select_frames/answer/evaluate cần có trên container vì pipeline.py gọi driver của chúng
# và chấm dev_300 ngay trên Modal (chạy detach).
LOCAL_SRC = ("config", "modal_infra", "methods", "subset", "evaluate", "select_frames", "answer")
ENV = {"HF_HOME": HF_PATH, "TOKENIZERS_PARALLELISM": "false"}

# Ghim phiên bản để kết quả tái lập được. Bộ này đã giải phụ thuộc chung với
# vllm==0.31.0 (torch==2.13.0, transformers<5.18, numpy<=2.3.x).
# Cả 3 image dùng chung COMMON_PKGS để numpy/opencv (đọc frame, ADA) giống nhau.
COMMON_PKGS = (
    "pandas==3.1.0", "pyarrow==26.0.0", "numpy==2.3.5",
    "opencv-python-headless==4.14.0.94", "pillow==13.0.0", "huggingface_hub==1.33.0",
)
TORCH_PKGS = ("torch==2.13.0", "torchvision==0.28.0", "transformers==5.17.0", "accelerate==1.15.0",
              "scikit-learn==1.7.2",   # KMeans cho KTV
              "sentencepiece==0.2.0")  # tokenizer SigLIP cho MDP3
VLLM_PKGS = ("vllm==0.31.0", "transformers==5.17.0")

# Image nhẹ: tải dữ liệu, chọn frame lấy đều
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install(*COMMON_PKGS)
    .env(ENV)
)

# Image chọn frame có mô hình chấm điểm (AKS, và các phương pháp khác cần torch)
selector_image = (
    base_image
    .uv_pip_install(*COMMON_PKGS, *TORCH_PKGS)
    .add_local_python_source(*LOCAL_SRC)
)

# Image chạy VLM trả lời câu hỏi
vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install(*COMMON_PKGS, *VLLM_PKGS)
    # debian_slim không có nvcc nên FlashInfer không JIT được kernel sampler top-k/top-p.
    # Dùng sampler PyTorch thay thế; temperature=0 (greedy) nên kết quả không đổi.
    # EngineCore phải được SPAWN, không fork: bộ xử lý ảnh InternVL2 chạy torch ở tiến trình
    # cha (warmup) làm khởi tạo pool OpenMP; fork xong, torch.zeros trong tiến trình con
    # kẹt vĩnh viễn trong libgomp (2026-11-19, py-spy --native, xem RUN_LOG).
    .env({**ENV, "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_python_source(*LOCAL_SRC)
)

base_image = base_image.add_local_python_source(*LOCAL_SRC)
