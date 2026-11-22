"""Baseline: lấy đều num_frames frame trên toàn video
(giống cách run.py gốc của UrbanVideo-Bench)."""
import numpy as np

from methods.base import FrameSelector, video_info


class UniformSelector(FrameSelector):
    name = "uniform"

    def select(self, video_path, question, num_frames, qid=None):
        total, _ = video_info(video_path)
        if total <= num_frames:
            return list(range(total))
        idx = np.linspace(0, total - 1, num_frames).round().astype(int)
        return sorted(set(idx.tolist()))
