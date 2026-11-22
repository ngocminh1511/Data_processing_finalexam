"""AKS — Adaptive Keyframe Sampling (Tang et al., CVPR 2025).

Quy trình cho mỗi câu hỏi:
  1. Lấy frame ứng viên (mặc định 1 frame/giây, xem base.candidate_frames)
  2. Chấm điểm câu hỏi–frame bằng CLIP/BLIP  (scorer.py) -> lưu cache
  3. Chọn frame bằng thuật toán ADA            (ada.py)
"""
import json
import os

from methods.aks.ada import ada_select
from methods.base import FrameSelector, candidate_frames, read_frames


class AKSSelector(FrameSelector):
    name = "aks"

    def __init__(self, scorer="clip", cand_fps=1.0, t1=0.8, t2=-100, all_depth=5,
                 cache_dir="cache/aks", **kwargs):
        super().__init__(**kwargs)
        self.scorer_kind = scorer
        self.cand_fps = cand_fps
        self.t1, self.t2, self.all_depth = t1, t2, all_depth
        self.cache_dir = os.path.join(cache_dir, scorer)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._scorer = None  # chỉ nạp mô hình khi thật sự cần chấm điểm

    def warmup(self):
        from methods.aks.scorer import Scorer
        if self._scorer is None:
            self._scorer = Scorer(self.scorer_kind)

    def _scores(self, video_path, question, num_frames, qid):
        path = os.path.join(self.cache_dir, f"{qid}.json") if qid is not None else None
        self.last_cache_hit = bool(path and os.path.exists(path))
        if self.last_cache_hit:
            with open(path) as f:
                c = json.load(f)
            return c["scores"], c["frames"]

        from methods.aks.scorer import question_only
        self.warmup()
        cand = candidate_frames(video_path, num_frames, self.cand_fps)
        frames = read_frames(video_path, cand)
        cand = [i for i in cand if i in frames]
        scores = self._scorer.score(question_only(question), [frames[i] for i in cand])

        if path:
            with open(path, "w") as f:
                json.dump({"scores": scores, "frames": cand}, f)
        return scores, cand

    def select(self, video_path, question, num_frames, qid=None):
        scores, cand = self._scores(video_path, question, num_frames, qid)
        return ada_select(scores, cand, num_frames, self.t1, self.t2, self.all_depth)
