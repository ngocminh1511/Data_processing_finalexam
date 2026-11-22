"""BƯỚC 0 — Tải dataset và trọng số model lên Volume của Modal (chạy 1 lần).

    modal run download.py                      # dataset + model chấm điểm + VLM mặc định
    modal run download.py --vlm Qwen/Qwen2.5-VL-3B-Instruct
    modal run download.py --skip-models        # chỉ tải dataset
    modal run download.py --vlm OpenGVLab/InternVL2-8B --scorers clip,dinov2   # AKS-CLIP + KTV

Dataset public (MIT) nên không cần HF token. Nếu bị giới hạn tốc độ, tạo secret
trên Modal tên "huggingface" chứa HF_TOKEN rồi bỏ comment dòng `secrets=` bên dưới.
"""
import modal

from config import DATA_DIR, DEFAULT_VLM, HF_DATASET, MCQ_PATH, SCORER_MODELS, VIDEO_DIR
from modal_infra import VOLUMES, base_image, hf_vol, vol

app = modal.App("uvb-download")


@app.function(image=base_image, volumes=VOLUMES, timeout=6 * 3600,
              # secrets=[modal.Secret.from_name("huggingface")],
              )
def download_dataset():
    import os

    import pandas as pd
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=HF_DATASET, repo_type="dataset", local_dir=DATA_DIR, max_workers=16)
    vol.commit()

    df = pd.read_parquet(MCQ_PATH)
    videos = set(os.listdir(VIDEO_DIR))
    missing = sorted(set(df["video_id"].astype(str)) - videos)
    print(f"Câu hỏi: {len(df)} | Video trên volume: {len(videos)} | Video thiếu: {len(missing)}")
    if missing:
        print("Ví dụ video thiếu:", missing[:5])
    print("Các cột:", list(df.columns))
    print(df["question_category"].value_counts().to_string())


@app.function(image=base_image, volumes=VOLUMES, timeout=3 * 3600)
def download_models(vlm: str, scorers: list[str]):
    from huggingface_hub import snapshot_download

    for repo in [*(SCORER_MODELS[s] for s in scorers), vlm]:
        print("Đang tải", repo)
        snapshot_download(repo_id=repo)
    hf_vol.commit()


@app.local_entrypoint()
def main(vlm: str = DEFAULT_VLM, skip_models: bool = False, skip_dataset: bool = False,
         scorers: str = "clip,blip"):
    calls = [] if skip_dataset else [download_dataset.spawn()]
    if not skip_models:
        calls.append(download_models.spawn(vlm, [s.strip() for s in scorers.split(",") if s.strip()]))
    for c in calls:
        c.get()
    print("Xong. Kiểm tra: modal volume ls urbanvideo-vol data/UrbanVideo-Bench")
