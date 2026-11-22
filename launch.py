"""Phát lệnh chạy một stage của pipeline.py mà KHÔNG phụ thuộc máy local.

    python launch.py night          # deploy app uvb-pipeline rồi spawn run_stage, thoát ngay
    python launch.py f8             # 8 frame: dò t1 trên dev_300 + 3 phương pháp trên subset_1000
    python launch.py blind          # baseline mù: 0 frame trên subset_1000
    python launch.py blind --model Qwen/Qwen2.5-VL-7B-Instruct
    python launch.py llava8         # LLaVA-NeXT-Video-7B: 0 frame + Uniform/AKS+E/KTV+E @8f
    python launch.py mdp3           # MDP3+E-32 trên InternVL2-8B, subset_1000
    python launch.py full --t1 0.3

Vì sao không dùng `modal run --detach`: khi tiến trình local bị tắt bình thường (đóng terminal,
kết thúc phiên), client Modal gửi tín hiệu HỦY cho hàm đang chạy -> run_stage bị hủy giữa chừng
(sự cố 2026-11-19, RUN_LOG). App đã deploy + spawn thì chạy độc lập với máy local.
Tiến độ: modal volume get urbanvideo-vol status/<stage>.json outputs/status/<stage>.json --force
"""
import argparse
import os
import subprocess
import sys

import modal

from subset import load_qids

APP = "uvb-pipeline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "night", "f8", "blind", "llava8", "mdp3", "full"])
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default=None, help="chỉ cho stage blind; mặc định InternVL2-8B")
    a = ap.parse_args()

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, "-m", "modal", "deploy", "pipeline.py"], check=True, env=env)

    subsets = {n: load_qids(f"outputs/subsets/{n}.txt") for n in ("dev_300", "subset_1000")}
    kw = {"model": a.model} if a.model else {}
    call = modal.Function.from_name(APP, "run_stage").spawn(a.stage, subsets, a.t1, a.limit, **kw)
    print(f"Đã gửi stage '{a.stage}' lên app đã deploy {APP}. Function call: {call.object_id}")
    model = a.model or ("llava-hf/LLaVA-NeXT-Video-7B-hf" if a.stage == "llava8" else None)
    st = a.stage + (f"__{model.split('/')[-1]}" if model else "")
    print("Có thể tắt máy. Xem tiến độ: "
          f"modal volume get urbanvideo-vol status/{st}.json outputs/status/{st}.json --force")


if __name__ == "__main__":
    main()
