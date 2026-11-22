"""BƯỚC 2 — Cho VLM trả lời câu hỏi, chỉ dùng các frame đã chọn ở bước 1.

    modal run answer.py --tag uniform_32f --qids outputs/subsets/subset_1000.txt
    modal run answer.py --tag aks-clip_32f --qids outputs/subsets/subset_1000.txt --shards 4
    modal run answer.py --tag aks-clip_32f --model Qwen/Qwen2.5-VL-3B-Instruct --gpu A10G
    modal run answer.py --tag aks-clip_32f --limit 50            # chạy thử
    modal run answer.py --tag uniform_32f --model OpenGVLab/InternVL2-8B --gpu H100 --batch 32
    modal run --detach answer.py ...                             # tắt máy vẫn chạy tiếp

Kết quả: /vol/results/<tag>__<model>__<subset>.csv và bản sao trong outputs/results/
         <subset> = tên file --qids (vd. subset_1000), head<N> khi dùng --limit, full khi chạy hết.
         /vol/results/parts/<tag>__<model>__<subset>/shard<k>of<n>.json: mỗi shard tự lưu khi xong;
         chạy lại cùng lệnh sẽ bỏ qua shard đã xong. Gộp + lưu nằm trong answer_driver trên Modal.

InternVL2: mỗi frame 1 tile 448 px (max_dynamic_patch=1) = 256 token, giống cách đưa frame
video của InternVL; max_model_len 16384 là đủ cho 32 frame.
"""
import json
import os

import modal

from config import (DEFAULT_VLM, MCQ_PATH, NUM_FRAMES, PROMPT, RESULT_DIR, SELECTION_DIR,
                    TEXT_ONLY_TAG, VIDEO_DIR)
from modal_infra import VOLUMES, vlm_image, vol

app = modal.App("uvb-answer")


def _encode(frame_bgr, size):
    """Thu nhỏ cạnh dài về `size` px rồi mã hóa JPEG base64 (giảm token ảnh)."""
    import base64

    import cv2
    h, w = frame_bgr.shape[:2]
    s = size / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


PREFETCH = 8  # số video đọc trước song song với suy luận (mỗi video 1 thread)


def is_video_llm(model: str) -> bool:
    """LLaVA-NeXT-Video: vLLM chỉ nhận MỘT video (mảng frame) mỗi prompt, không nhận ảnh rời."""
    return "llava-next-video" in model.lower()


def _rgb(frame_bgr, size):
    """Frame cho Video-LLM: BGR -> RGB, thu nhỏ cạnh dài về `size` px (processor tự resize tiếp)."""
    import cv2
    h, w = frame_bgr.shape[:2]
    s = size / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def llava_prompt(text: str, has_video: bool) -> str:
    """Định dạng hội thoại của llava-hf/LLaVA-NeXT-Video-7B-hf (giống ví dụ offline của vLLM)."""
    return f"USER: <video>\n{text} ASSISTANT:" if has_video else f"USER: {text} ASSISTANT:"


def llm_kwargs(model: str, max_model_len: int, max_images: int) -> dict:
    """Tham số riêng theo họ model cho vllm.LLM."""
    m = model.lower()
    if "internvl" in m:
        return {"max_model_len": min(max_model_len, 16384), "limit_mm_per_prompt": {"image": max_images},
                "mm_processor_kwargs": {"max_dynamic_patch": 1}}
    if is_video_llm(model):
        # Vicuna-7B: ngữ cảnh 4096; mỗi frame 144 token -> 8 frame ~1,2k token
        return {"max_model_len": min(max_model_len, 4096), "limit_mm_per_prompt": {"video": 1}}
    return {"max_model_len": max_model_len, "limit_mm_per_prompt": {"image": max_images}}


@app.cls(image=vlm_image, volumes=VOLUMES, gpu="L40S", cpu=PREFETCH, timeout=3 * 3600,
         startup_timeout=15 * 60,  # @enter (nạp model + khởi tạo engine) treo quá 15 phút -> lỗi
         scaledown_window=120)
class VLM:
    model: str = modal.parameter(default=DEFAULT_VLM)
    max_images: int = modal.parameter(default=NUM_FRAMES)
    max_model_len: int = modal.parameter(default=32768)

    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=self.model, gpu_memory_utilization=0.9, trust_remote_code=True,
                       **llm_kwargs(self.model, self.max_model_len, self.max_images))
        self.video_llm = is_video_llm(self.model)
        self.sp = SamplingParams(temperature=0, max_tokens=256)

    @modal.method()
    def run(self, tag, shard, n_shards, limit, qids, frame_size, batch, part_path=None):
        import time
        from collections import defaultdict, deque
        from concurrent.futures import ThreadPoolExecutor

        import pandas as pd

        from methods.base import read_frames
        from subset import filter_mcq, shard_rows

        vol.reload()
        text_only = tag == TEXT_ONLY_TAG  # baseline mù: không đưa frame nào, giữ nguyên prompt
        if text_only:
            sel = {}
        else:
            with open(f"{SELECTION_DIR}/{tag}.json") as f:
                sel = json.load(f)
        df = filter_mcq(pd.read_parquet(MCQ_PATH), limit, qids)
        rows = [r for r in shard_rows(df, shard, n_shards)
                if text_only or sel.get(str(r["Question_id"]))]

        # gom câu hỏi theo video để mỗi video chỉ đọc 1 lần
        by_video = defaultdict(list)
        for r in rows:
            by_video[str(r["video_id"])].append(r)

        results, convs, meta = [], [], []

        def flush():
            if not convs:
                return
            t0 = time.perf_counter()
            if self.video_llm:  # convs là prompt dict {"prompt", "multi_modal_data"}
                outs = self.llm.generate(convs, self.sp, use_tqdm=False)
            else:
                outs = self.llm.chat(convs, self.sp, use_tqdm=False)
            dt = (time.perf_counter() - t0) / len(convs)  # thời gian suy luận chia đều trong batch
            for (qid, n), o in zip(meta, outs):
                results.append({"Question_id": qid, "Output": o.outputs[0].text, "n_frames": n,
                                "infer_s": round(dt, 4)})
            convs.clear()
            meta.clear()

        def load_video(qs):
            """Đọc + mã hóa các frame cần của một video (chạy trong thread; cv2 nhả GIL)."""
            need = set()
            for r in qs:
                need.update(sel.get(str(r["Question_id"]), [])[: self.max_images])
            if not need:
                return {}
            frames = read_frames(os.path.join(VIDEO_DIR, str(qs[0]["video_id"])), need)
            if self.video_llm:
                return {i: _rgb(f, frame_size) for i, f in frames.items()}
            return {i: _encode(f, frame_size) for i, f in frames.items()}

        # Đọc trước PREFETCH video trong lúc GPU chạy batch hiện tại. Thứ tự video,
        # frame và nội dung message giữ nguyên như đọc tuần tự, nên kết quả không đổi.
        groups = list(by_video.values())
        with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
            pending = deque(pool.submit(load_video, qs) for qs in groups[:PREFETCH])
            for k, qs in enumerate(groups):
                enc = pending.popleft().result()
                if k + PREFETCH < len(groups):
                    pending.append(pool.submit(load_video, groups[k + PREFETCH]))
                for r in qs:
                    qid = str(r["Question_id"])
                    idx = [i for i in sel.get(qid, [])[: self.max_images] if i in enc]
                    text = PROMPT + "\n" + r["question"]
                    if self.video_llm:
                        item = {"prompt": llava_prompt(text, bool(idx))}
                        if idx:
                            import numpy as np
                            # cùng kích thước -> xếp thành video (T, H, W, 3) theo thứ tự thời gian
                            item["multi_modal_data"] = {"video": np.stack([enc[i] for i in idx])}
                        convs.append(item)
                    else:
                        content = [{"type": "text", "text": text}]
                        content += [{"type": "image_url", "image_url": {"url": enc[i]}} for i in idx]
                        convs.append([{"role": "user", "content": content}])
                    meta.append((qid, len(idx)))
                    if len(convs) >= batch:
                        flush()
                if k % 50 == 0:
                    print(f"[shard {shard}] video {k}/{len(by_video)}, đã trả lời {len(results)}")
        flush()
        if part_path:
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            with open(part_path, "w") as f:
                json.dump(results, f)
            vol.commit()
        return results


def result_name(tag: str, model: str, subset: str) -> str:
    return f"{tag}__{model.split('/')[-1]}__{subset}"


@app.function(image=vlm_image, volumes=VOLUMES, timeout=24 * 3600)
def answer_driver(tag, model, gpu, shards, limit, qids, subset, frame_size=448, batch=16,
                  max_images=NUM_FRAMES):
    """Chạy toàn bộ bước trả lời trên Modal: chia shard, bỏ qua shard đã xong, gộp, lưu CSV.
    Trả về (tên file, records) để bản local lưu thêm một bản sao."""
    import pandas as pd

    from subset import filter_mcq

    name = result_name(tag, model, subset)
    part_dir = f"{RESULT_DIR}/parts/{name}"
    vol.reload()
    answers, todo = [], []
    for s in range(shards):
        p = f"{part_dir}/shard{s}of{shards}.json"
        if os.path.exists(p):
            with open(p) as f:
                answers += json.load(f)
        else:
            todo.append((tag, s, shards, limit, qids, frame_size, batch, p))
    print(f"[{name}] {shards - len(todo)}/{shards} shard đã xong từ trước, chạy {len(todo)} shard")
    if todo:
        vlm = VLM.with_options(gpu=gpu)(model=model, max_images=max_images)
        for part in vlm.run.starmap(todo):
            answers += part

    mcq = filter_mcq(pd.read_parquet(MCQ_PATH), limit, qids)
    mcq["Question_id"] = mcq["Question_id"].astype(str)
    out = mcq.merge(pd.DataFrame(answers), on="Question_id", how="left")
    out.insert(0, "tag", tag)
    out.insert(1, "model", model)
    out.insert(2, "subset", subset)
    out.insert(3, "gpu", gpu)

    vol.reload()
    os.makedirs(RESULT_DIR, exist_ok=True)
    out.to_csv(f"{RESULT_DIR}/{name}.csv", index=False)
    vol.commit()
    print(f"Đã trả lời {out['Output'].notna().sum()}/{len(out)} câu -> {RESULT_DIR}/{name}.csv")
    return name, out.to_dict("records")


@app.local_entrypoint()
def main(tag: str, model: str = DEFAULT_VLM, gpu: str = "L40S", shards: int = 1,
         limit: int = 0, qids: str = "", frame_size: int = 448, batch: int = 16,
         max_images: int = NUM_FRAMES):
    import pandas as pd

    from subset import load_qids, run_name

    qid_list = load_qids(qids) if qids else None
    name, records = answer_driver.remote(tag, model, gpu, shards, limit, qid_list,
                                         run_name(qids, limit), frame_size, batch, max_images)
    out = pd.DataFrame(records)
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv(f"outputs/results/{name}.csv", index=False)

    print(f"Đã trả lời {out['Output'].notna().sum()}/{len(out)} câu -> outputs/results/{name}.csv")
    print("Bước tiếp theo: python evaluate.py")
