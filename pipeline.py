"""Chạy cả chuỗi chọn frame -> trả lời trên Modal, điều phối bằng một hàm chạy trên Modal,
nên dùng được --detach (tắt máy / ngủ vẫn chạy tiếp, kết quả nằm trên Volume).

    modal run pipeline.py --stage smoke                 # chạy thử 20 câu (nên chờ xem kết quả)
    modal run --detach pipeline.py --stage night        # dò t1 trên dev_300 + 3 phương pháp trên subset_1000
    modal run --detach pipeline.py --stage full         # 3 phương pháp trên toàn bộ 5.355 câu
    python launch.py f8                                 # như night nhưng 8 frame (dò t1 lại ở 8 frame)
    python launch.py blind                              # baseline mù: 0 frame trên subset_1000
    python launch.py blind --model Qwen/Qwen2.5-VL-7B-Instruct   # baseline mù cho mô hình khác
    python launch.py llava8                             # LLaVA-NeXT-Video: 0 frame + 3 phương pháp @8f
    python launch.py mdp3                               # MDP3+E-32 trên InternVL2-8B, subset_1000
    (nên phát lệnh bằng launch.py: deploy + spawn, không phụ thuộc máy local)
    modal volume get urbanvideo-vol status outputs/ --force   # xem tiến độ
    modal volume get urbanvideo-vol results outputs/ --force  # tải kết quả rồi: python evaluate.py

Phương pháp (cùng 32 frame, cùng InternVL2-8B):
  Uniform-32 | AKS+E-32 (CLIP, t1 chọn trên dev_300) | KTV+E-32   (+E = luôn có frame đầu/cuối)
Quy tắc chọn t1 (ghi trước khi chạy): Overall cao nhất trên dev_300; hòa thì lấy t1 lớn hơn.
Tiến độ: /vol/status/<stage>.json sau mỗi bước.
"""
import json
import os
import time

import modal

import answer
import select_frames
from config import NUM_FRAMES, TEXT_ONLY_TAG, VOL_PATH, selection_tag
from modal_infra import VOLUMES, base_image, vol

app = modal.App("uvb-pipeline")
app.include(select_frames.app)
app.include(answer.app)

MODEL = "OpenGVLab/InternVL2-8B"
LLAVA = "llava-hf/LLaVA-NeXT-Video-7B-hf"  # Video-LLM chuyên video, có trong Table 3 (38,6 @ 32f)
LLAVA_T1 = 0.3  # t1 của AKS+E-8 đã chọn trên dev_300 với InternVL2 (stage f8), dùng lại
GPU, BATCH = "H100", 32
T1S = (0.8, 0.3, 0.2)
STATUS_DIR = f"{VOL_PATH}/status"
UNIFORM = ("uniform", {})
KTV = ("ktv", {"endpoints": True})
# 8 frame: KTV dùng tập ứng viên của lượt 32 frame (cand_k=32) -> cùng ứng viên với AKS
# (điểm CLIP đã cache theo tập 32) và dùng lại cache feature DINOv2, không tính lại.
KTV8 = ("ktv", {"endpoints": True, "cand_k": 32})
MDP3 = ("mdp3", {"endpoints": True})


def aks(t1, nf=NUM_FRAMES):
    p = {"scorer": "clip", "t1": t1, "endpoints": True}
    if nf < 32:
        # ADA lấy k = int(K / 2^depth) frame mỗi đoạn: với K = 8 phải dừng ở độ sâu log2(8) = 3,
        # all_depth = 5 cho 0 frame (đã kiểm tra trên cache: t1 = 0.8 -> 0 frame ở 100% câu).
        p["all_depth"] = max(1, nf.bit_length() - 1)
    return ("aks", p)


TEXT_ONLY = ("textonly", {})  # baseline mù: 0 frame


def tag_of(method, nf=NUM_FRAMES):
    if method == TEXT_ONLY:
        return TEXT_ONLY_TAG
    return selection_tag(method[0], nf, method[1])


def score(records):
    """Overall, unparsed, số frame TB của một lượt trả lời."""
    from evaluate import extract_option
    pred = [extract_option(r.get("Output")) for r in records]
    gt = [extract_option(r.get("answer")) for r in records]
    nf = [r["n_frames"] for r in records if r.get("n_frames") == r.get("n_frames") and r.get("n_frames")]
    return {"n": len(records), "answered": sum(1 for r in records if isinstance(r.get("Output"), str)),
            "unparsed": sum(p is None for p in pred),
            "acc": round(100 * sum(p == g and p is not None for p, g in zip(pred, gt)) / max(len(records), 1), 2),
            "mean_frames": round(sum(nf) / max(len(nf), 1), 2)}


class Status:
    def __init__(self, stage):
        self.path = f"{STATUS_DIR}/{stage}.json"
        self.d = {"stage": stage, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": [], "done": False}
        self.t0 = time.time()

    def log(self, name, **info):
        self.d["steps"].append({"step": name, "elapsed_s": round(time.time() - self.t0), **info})
        print(f"[{self.d['stage']}] {name}: {info}")
        self.save()

    def save(self):
        vol.reload()
        os.makedirs(STATUS_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=1, ensure_ascii=False)
        vol.commit()


def select_many(st, methods, qids, run, limit=0, shards=None, nf=NUM_FRAMES):
    """Chọn frame song song cho nhiều phương pháp (mỗi phương pháp một driver)."""
    calls = []
    for m in methods:
        n = shards or (8 if m[0] == "ktv" else 2)
        calls.append((m, select_frames.select_driver.spawn(m[0], nf, m[1], n, limit, qids,
                                                           tag_of(m, nf), run)))
    for m, c in calls:
        _, _, info = c.get()
        st.log(f"select {tag_of(m, nf)}", **info)
        if info["failed"]:
            raise RuntimeError(f"{tag_of(m, nf)}: {info['failed']} câu chọn frame lỗi")


def answer_many(st, methods, qids, subset, limit=0, shards=1, nf=NUM_FRAMES, model=MODEL):
    """Trả lời song song cho nhiều phương pháp; trả về {tag: điểm}."""
    calls = [(m, answer.answer_driver.spawn(tag_of(m, nf), model, GPU, shards, limit, qids, subset,
                                            448, BATCH, max(nf, 1))) for m in methods]
    out = {}
    for m, c in calls:
        name, records = c.get()
        out[tag_of(m, nf)] = score(records)
        st.log(f"answer {name}", **out[tag_of(m, nf)])
    return out


@app.function(image=base_image, volumes=VOLUMES, timeout=24 * 3600)
def run_stage(stage: str, subsets: dict, t1: float | None = None, limit: int = 20,
              model: str = MODEL):
    if stage == "llava8" and model == MODEL:
        model = LLAVA
    # model khác mặc định: file tiến độ có hậu tố tên model
    st = Status(stage if model == MODEL else f"{stage}__{model.split('/')[-1]}")
    try:
        if stage == "smoke":
            methods = [UNIFORM, aks(0.3), KTV]
            select_many(st, methods, None, f"head{limit}", limit=limit, shards=1)
            res = answer_many(st, methods, None, f"head{limit}", limit=limit, shards=1)
            bad = {k: v for k, v in res.items()
                   if v["answered"] < v["n"] or v["unparsed"] or v["mean_frames"] < 31}
            st.d["ok"] = not bad
            if bad:
                st.log("KIỂM TRA KHÔNG ĐẠT", bad=bad)

        elif stage == "night":
            dev, test = subsets["dev_300"], subsets["subset_1000"]
            both = sorted(set(dev) | set(test), key=int)
            select_many(st, [UNIFORM, KTV, *(aks(t) for t in T1S)], both, "dev_300+subset_1000")

            dev_res = answer_many(st, [aks(t) for t in T1S], dev, "dev_300", shards=1)
            best = max(T1S, key=lambda t: (dev_res[tag_of(aks(t))]["acc"], t))
            st.d["chosen_t1"] = best
            st.log("chọn t1", t1=best, dev={t: dev_res[tag_of(aks(t))]["acc"] for t in T1S})

            answer_many(st, [UNIFORM, aks(best), KTV], test, "subset_1000", shards=4)

        elif stage == "f8":
            # Cùng quy trình như night nhưng 8 frame; t1 dò lại ở 8 frame (cùng quy tắc ghi trước).
            nf = 8
            dev, test = subsets["dev_300"], subsets["subset_1000"]
            both = sorted(set(dev) | set(test), key=int)
            select_many(st, [UNIFORM, KTV8, *(aks(t, nf) for t in T1S)], both,
                        "dev_300+subset_1000", nf=nf)

            dev_res = answer_many(st, [aks(t, nf) for t in T1S], dev, "dev_300", shards=1, nf=nf)
            best = max(T1S, key=lambda t: (dev_res[tag_of(aks(t, nf), nf)]["acc"], t))
            st.d["chosen_t1"] = best
            st.log("chọn t1", t1=best, dev={t: dev_res[tag_of(aks(t, nf), nf)]["acc"] for t in T1S})

            answer_many(st, [UNIFORM, aks(best, nf), KTV8], test, "subset_1000", shards=2, nf=nf)

        elif stage == "blind":
            # Baseline mù trên subset_1000: cùng prompt + câu hỏi, 0 frame. Đo mức mô hình dựa
            # vào video: nếu gần bằng Uniform thì chọn frame khó tạo khác biệt.
            answer_many(st, [TEXT_ONLY], subsets["subset_1000"], "subset_1000", shards=2, nf=0,
                        model=model)

        elif stage == "llava8":
            # LLaVA-NeXT-Video-7B, 8 frame, subset_1000: 0 frame + 3 phương pháp. Dùng lại đúng các
            # file chọn frame 8f của stage f8 (không chọn lại); t1 của AKS lấy từ f8 (dò trên InternVL2).
            test = subsets["subset_1000"]
            answer_many(st, [TEXT_ONLY], test, "subset_1000", shards=1, nf=0, model=model)
            answer_many(st, [UNIFORM, aks(LLAVA_T1, 8), KTV8], test, "subset_1000", shards=2, nf=8,
                        model=model)

        elif stage == "mdp3":
            # MDP3+E-32 (ICCV 2025) trên InternVL2-8B, subset_1000; tham số mặc định của code gốc
            # (lambda 0.2, đoạn 32 frame), không dò trên dev. So với Uniform/AKS+E/KTV+E-32 của night.
            test = subsets["subset_1000"]
            select_many(st, [MDP3], test, "subset_1000", shards=8)
            answer_many(st, [MDP3], test, "subset_1000", shards=4)

        elif stage == "full":
            if t1 is None:
                with open(f"{STATUS_DIR}/night.json") as f:
                    t1 = json.load(f)["chosen_t1"]
            methods = [UNIFORM, aks(t1), KTV]
            select_many(st, methods, None, "full", shards=None)
            answer_many(st, methods, None, "full", shards=8)
        else:
            raise ValueError(f"stage không hợp lệ: {stage}")
        st.d["done"] = True
        st.log("XONG")
    except Exception as e:
        st.d["error"] = repr(e)
        st.log("LỖI", error=repr(e))
        raise
    return st.d


@app.local_entrypoint()
def main(stage: str = "smoke", t1: float = -1.0, limit: int = 20):
    from subset import load_qids

    subsets = {n: load_qids(f"outputs/subsets/{n}.txt") for n in ("dev_300", "subset_1000")}
    res = run_stage.remote(stage, subsets, None if t1 < 0 else t1, limit)
    print(json.dumps(res, indent=1, ensure_ascii=False))
