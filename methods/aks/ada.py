"""Thuật toán ADA của AKS (judge-and-split), tách từ frame_select.py
của repo gốc ncTimTang/AKS. Chỉ dùng numpy, không phụ thuộc mô hình.

Ý tưởng: xét một đoạn video. Nếu các frame điểm cao nhất nổi bật hẳn so với
trung bình (mean_diff > t1) thì lấy top-k trong đoạn đó (ưu tiên độ liên quan).
Nếu không thì chia đôi đoạn và xét tiếp (ưu tiên độ phủ thời gian),
tối đa all_depth lần chia.
"""
import heapq

import numpy as np


def _judge_and_split(segments, n, t1, t2, all_depth):
    keep, split = [], []
    for seg in segments:
        score, depth = seg["score"], seg["depth"]
        top = heapq.nlargest(n, range(len(score)), score.__getitem__)
        mean_diff = np.mean([score[t] for t in top]) - np.mean(score)
        if mean_diff > t1 and np.std(score) > t2:
            keep.append(seg)
        elif depth < all_depth and len(score) >= 2:
            h = len(score) // 2
            split.append(dict(score=score[:h], fn=seg["fn"][:h], depth=depth + 1))
            split.append(dict(score=score[h:], fn=seg["fn"][h:], depth=depth + 1))
        else:
            keep.append(seg)
    if split:
        keep += _judge_and_split(split, n, t1, t2, all_depth)
    return keep


def ada_select(scores, frame_ids, num_frames, t1=0.8, t2=-100, all_depth=5):
    """scores    : điểm liên quan câu hỏi–frame của các frame ứng viên
    frame_ids : chỉ số frame tương ứng trong video
    Trả về danh sách chỉ số frame được chọn, sắp xếp tăng dần."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= num_frames:
        return sorted(int(f) for f in frame_ids)

    rng = scores.max() - scores.min()
    norm = (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)

    segments = _judge_and_split(
        [dict(score=norm, fn=list(frame_ids), depth=0)],
        num_frames, t1, t2, all_depth,
    )

    out = []
    for seg in segments:
        k = int(num_frames / 2 ** seg["depth"])
        top = heapq.nlargest(k, range(len(seg["score"])), seg["score"].__getitem__)
        out.extend(int(seg["fn"][t]) for t in top)
    return sorted(set(out))
