"""BƯỚC 1 — Chọn frame cho từng câu hỏi bằng một phương pháp trong methods/.

    modal run select_frames.py --method uniform --qids outputs/subsets/subset_1000.txt
    modal run select_frames.py --method aks --params '{"scorer": "clip"}' --qids outputs/subsets/subset_1000.txt
    modal run select_frames.py --method aks --params '{"scorer": "clip", "t1": 0.3, "endpoints": true}'
    modal run select_frames.py --method ktv --params '{"endpoints": true}' --shards 8
    modal run select_frames.py --method aks --limit 50          # chạy thử 50 câu
    modal run --detach select_frames.py ...                    # tắt máy vẫn chạy tiếp

Tham số chung: "endpoints": true = biến thể +E (luôn có frame đầu + cuối, tổng vẫn num_frames).

Kết quả: /vol/selections/<tag>.json         ({Question_id: [chỉ số frame]})
         /vol/selections/<tag>.timing.json  ({Question_id: {"s": giây, "cache_hit": bool|null}})
         /vol/selections/parts/<tag>/<run>/shard<k>of<n>.json  (mỗi shard tự lưu khi xong;
            chạy lại cùng lệnh sẽ bỏ qua shard đã xong)
         và bản sao ở outputs/selections/ trên máy bạn (nếu tiến trình local còn sống).
Toàn bộ việc gộp + lưu nằm trong select_driver trên Modal, nên chạy được với --detach.
Kết quả mới được GỘP vào file cũ cùng tag (cùng tag = cùng tham số nên chọn frame như nhau).
"""
import json
import os

import modal

from config import CACHE_DIR, MCQ_PATH, NUM_FRAMES, SELECTION_DIR, VIDEO_DIR, selection_tag
from modal_infra import VOLUMES, base_image, selector_image, vol

app = modal.App("uvb-select")

CPU_METHODS = {"uniform"}  # phương pháp không cần GPU
SHARD_TIMEOUT = 3 * 3600   # chặn trường hợp chạy mất kiểm soát


def _run_shard(method, num_frames, params, shard, n_shards, limit, qids, part_path=None):
    import time

    import pandas as pd

    from methods import get_method
    from methods.base import add_endpoints, video_info
    from subset import filter_mcq, shard_rows

    df = filter_mcq(pd.read_parquet(MCQ_PATH), limit, qids)
    rows = shard_rows(df, shard, n_shards)  # chia theo video

    params = {"cache_dir": f"{CACHE_DIR}/{method}", **params}
    endpoints = params.pop("endpoints", False)
    selector = get_method(method, **params)
    selector.warmup()  # nạp mô hình chấm điểm ngoài phần bấm giờ

    out, times = {}, {}
    for k, r in enumerate(rows):
        qid = str(r["Question_id"])
        path = os.path.join(VIDEO_DIR, str(r["video_id"]))
        t0 = time.perf_counter()
        try:
            sel = selector.select(path, r["question"], num_frames, qid=qid)
            if endpoints and sel:
                sel = add_endpoints(sel, video_info(path)[0], num_frames)
            out[qid] = sel
        except Exception as e:  # một video lỗi không làm hỏng cả lượt chạy
            print(f"[shard {shard}] lỗi câu {qid}: {e}")
            out[qid] = None
        times[qid] = {"s": round(time.perf_counter() - t0, 4), "cache_hit": selector.last_cache_hit}
        if k % 100 == 0:
            print(f"[shard {shard}] {k}/{len(rows)}")
            vol.commit()  # lưu cache định kỳ
    if part_path:
        os.makedirs(os.path.dirname(part_path), exist_ok=True)
        with open(part_path, "w") as f:
            json.dump({"out": out, "times": times}, f)
    vol.commit()
    return out, times


@app.function(image=base_image, volumes=VOLUMES, timeout=SHARD_TIMEOUT)
def select_cpu(method, num_frames, params, shard, n_shards, limit, qids, part_path=None):
    return _run_shard(method, num_frames, params, shard, n_shards, limit, qids, part_path)


@app.function(image=selector_image, volumes=VOLUMES, gpu="T4", timeout=SHARD_TIMEOUT)
def select_gpu(method, num_frames, params, shard, n_shards, limit, qids, part_path=None):
    return _run_shard(method, num_frames, params, shard, n_shards, limit, qids, part_path)


def _merge_json(path, data):
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    old.update(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(old, f)
    return old


@app.function(image=base_image, volumes=VOLUMES, timeout=24 * 3600)
def select_driver(method, num_frames, params, shards, limit, qids, tag, run, gpu="T4"):
    """Chạy toàn bộ bước chọn frame trên Modal: chia shard, bỏ qua shard đã xong, gộp, lưu."""
    fn = select_cpu if method in CPU_METHODS else (
        select_gpu if gpu == "T4" else select_gpu.with_options(gpu=gpu))
    part_dir = f"{SELECTION_DIR}/parts/{tag}/{run}"
    vol.reload()
    merged, times, todo = {}, {}, []
    for s in range(shards):
        p = f"{part_dir}/shard{s}of{shards}.json"
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            merged.update(d["out"])
            times.update(d["times"])
        else:
            todo.append((method, num_frames, params, s, shards, limit, qids, p))
    print(f"[{tag}/{run}] {shards - len(todo)}/{shards} shard đã xong từ trước, chạy {len(todo)} shard")
    if todo:  # starmap([]) của Modal chờ mãi, không trả về
        for out, t in fn.starmap(todo):
            merged.update(out)
            times.update(t)

    vol.reload()
    total = len(_merge_json(f"{SELECTION_DIR}/{tag}.json", merged))
    _merge_json(f"{SELECTION_DIR}/{tag}.timing.json", times)
    vol.commit()

    n = [len(v) for v in merged.values() if v]
    fresh = [t["s"] for t in times.values() if t["cache_hit"] is not True]
    info = {"tag": tag, "run": run, "questions": len(merged), "in_file": total,
            "failed": sum(1 for v in merged.values() if not v),
            "mean_frames": round(sum(n) / max(len(n), 1), 2),
            "select_s": round(sum(fresh) / max(len(fresh), 1), 3), "fresh": len(fresh)}
    print(info)
    return merged, times, info


@app.local_entrypoint()
def main(method: str = "aks", num_frames: int = NUM_FRAMES, params: str = "{}",
         shards: int = 4, limit: int = 0, qids: str = "", tag: str = "", gpu: str = "T4"):
    from subset import load_qids, run_name

    p = json.loads(params)
    if method == "aks":
        p.setdefault("scorer", "clip")
    tag = tag or selection_tag(method, num_frames, p)
    qid_list = load_qids(qids) if qids else None

    merged, times, info = select_driver.remote(method, num_frames, p, shards, limit, qid_list,
                                               tag, run_name(qids, limit), gpu)
    _merge_json(f"outputs/selections/{tag}.json", merged)
    _merge_json(f"outputs/selections/{tag}.timing.json", times)

    print(f"Tag: {tag} | {info['questions']} câu lượt này ({info['in_file']} câu trong file) | "
          f"lỗi: {info['failed']} | số frame TB: {info['mean_frames']:.1f} | "
          f"chọn frame TB: {info['select_s']:.2f} s/câu (không tính cache, n={info['fresh']})")
    print(f"Bước tiếp theo: modal run answer.py --tag {tag}" + (f" --qids {qids}" if qids else ""))
