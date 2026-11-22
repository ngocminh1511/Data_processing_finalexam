"""KTV giai đoạn 1 — chọn keyframe bằng phân cụm feature DINOv2 (Song et al., AAAI 2026).

Port từ repo gốc github.com/hktk07/KTV (keyframe_select_new.py, cluster_keyframe_and_order.py):
  - Feature: facebook/dinov2-large, pooler_output (token CLS), không chuẩn hóa L2.
    Tiền xử lý: Resize(256, bicubic) -> CenterCrop(224) -> chuẩn hóa ImageNet.
  - KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0) trên feature thô.
  - Mỗi tâm cụm lấy frame gần nhất (khoảng cách Euclid, xét mọi frame ứng viên).

Khác bài gốc (ghi trong docs/giao-thuc-danh-gia-urbanvideo.md):
  1. Ứng viên = candidate_frames() (1 fps, tối thiểu 2k) giống AKS, thay vì tối đa 5400 frame.
  2. Hai tâm cụm trùng frame gần nhất -> lấy frame gần kế tiếp chưa dùng
     (bài gốc: bỏ hết, quay về lấy đều).
  3. Trả về theo thứ tự thời gian, bỏ bước xếp lại bằng CLIP (VLM nhận frame theo thời gian).
  4. Chỉ giai đoạn 1 (chọn frame), không cắt token.
Không phụ thuộc câu hỏi nên feature được cache theo video.
"""
import os

import numpy as np

from methods.base import FrameSelector, candidate_frames


def ktv_select(feats, frame_ids, k: int, seed: int = 0) -> list[int]:
    """feats: (n, d) feature của các frame ứng viên; frame_ids: chỉ số frame tương ứng."""
    frame_ids = [int(f) for f in frame_ids]
    if len(frame_ids) <= k:
        return sorted(frame_ids)
    from sklearn.cluster import KMeans

    feats = np.asarray(feats, dtype=np.float32)
    km = KMeans(n_clusters=k, random_state=seed, init="k-means++", n_init=10).fit(feats)
    dist = np.linalg.norm(feats[None, :, :] - km.cluster_centers_[:, None, :], axis=2)  # (k, n)
    used, picked = set(), []
    for c in range(k):
        for j in np.argsort(dist[c], kind="stable"):
            if int(j) not in used:
                used.add(int(j))
                picked.append(int(j))
                break
    return sorted(frame_ids[j] for j in picked)


class KTVSelector(FrameSelector):
    name = "ktv"

    def __init__(self, cand_fps=1.0, model="facebook/dinov2-large", batch=64,
                 cache_dir="cache/ktv", cand_k=None, **kwargs):
        """cand_k: số frame dùng để tạo tập ứng viên (mặc định = num_frames). Đặt 32 khi chọn
        8 frame để dùng chung tập ứng viên (và cache feature) với lượt 32 frame và với AKS."""
        super().__init__(**kwargs)
        self.cand_fps, self.model_name, self.batch = cand_fps, model, batch
        self.cand_k = cand_k
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._model = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from transformers import Dinov2Model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = Dinov2Model.from_pretrained(self.model_name).to(self.device).eval()
        self._tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract(self, video_path, indices):
        """Đọc tuần tự video, biến đổi ngay từng frame cần (không giữ ảnh gốc trong RAM)."""
        import cv2
        import torch
        from PIL import Image

        wanted, last = set(indices), max(indices)
        tensors, kept = [], []
        cap = cv2.VideoCapture(video_path)
        i = 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                tensors.append(self._tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                kept.append(i)
            i += 1
        cap.release()
        feats = []
        with torch.no_grad():
            for s in range(0, len(tensors), self.batch):
                x = torch.stack(tensors[s:s + self.batch]).to(self.device)
                feats.append(self._model(pixel_values=x).pooler_output.float().cpu().numpy())
        return np.concatenate(feats), kept

    def _features(self, video_path, num_frames):
        num_frames = self.cand_k or num_frames  # tập ứng viên, không phải số frame chọn
        key = f"{os.path.basename(video_path)}_k{num_frames}.npz"
        path = os.path.join(self.cache_dir, key)
        self.last_cache_hit = os.path.exists(path)
        if self.last_cache_hit:
            z = np.load(path)
            return z["feats"], z["frames"].tolist()
        self.warmup()
        feats, kept = self._extract(video_path, candidate_frames(video_path, num_frames, self.cand_fps))
        np.savez(path, feats=feats, frames=np.asarray(kept))
        return feats, kept

    def select(self, video_path, question, num_frames, qid=None):
        feats, cand = self._features(video_path, num_frames)
        return ktv_select(feats, cand, num_frames)  # k = num_frames cụm
