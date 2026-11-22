"""Giao diện chung cho mọi phương pháp chọn frame.

Mỗi phương pháp chỉ cần kế thừa FrameSelector và cài đặt hàm select().
select_frames.py không cần biết bên trong phương pháp làm gì.
"""
from abc import ABC, abstractmethod

import cv2


class FrameSelector(ABC):
    name = "base"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    @abstractmethod
    def select(self, video_path: str, question: str, num_frames: int, qid=None) -> list[int]:
        """Trả về danh sách chỉ số frame (đã sắp xếp tăng dần) sẽ đưa vào VLM.

        video_path : đường dẫn file video
        question   : nội dung câu hỏi (đã bao gồm các lựa chọn A-E)
        num_frames : số frame tối đa được chọn
        qid        : Question_id, dùng làm khóa cache nếu phương pháp cần
        """

    def warmup(self):
        """Nạp mô hình trước khi bấm giờ (để thời gian chọn frame không gồm lần nạp)."""

    # Phương pháp có cache gán True/False sau mỗi select(); None = không có cache
    last_cache_hit = None


def candidate_frames(video_path: str, num_frames: int, cand_fps: float = 1.0) -> list[int]:
    """Tập frame ứng viên dùng chung cho AKS, KTV (giao thức mục 4.2).

    Lấy đều duration × cand_fps frame; tối thiểu 2 × num_frames, không vượt tổng số frame.
    """
    import numpy as np
    total, fps = video_info(video_path)
    n = int(total / fps * cand_fps)
    n = min(total, max(n, 2 * num_frames))  # video ngắn: vẫn đủ ứng viên
    return sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))


def add_endpoints(sel, total: int, num_frames: int) -> list[int]:
    """Biến thể +E: luôn có frame đầu (0) và frame cuối (total-1), tổng vẫn ≤ num_frames.

    Phương pháp chọn đủ num_frames như bình thường; thêm 2 frame đầu/cuối, rồi chừng nào
    còn thừa thì bỏ frame ở giữa "thừa" nhất = frame có khoảng cách tới frame kề gần nhất
    nhỏ nhất (hòa thì bỏ frame sớm hơn). Không chạy phương pháp với num_frames-2 vì ADA
    với K=30 cho k = int(30/32) = 0 frame ở độ sâu 5.
    """
    if total <= 0:
        return sorted(set(int(i) for i in sel))
    out = sorted(set(int(i) for i in sel) | {0, total - 1})
    while len(out) > num_frames and len(out) > 2:
        gaps = [min(out[i] - out[i - 1], out[i + 1] - out[i]) for i in range(1, len(out) - 1)]
        del out[1 + min(range(len(gaps)), key=gaps.__getitem__)]
    return out


def video_info(video_path: str):
    """Trả về (tổng số frame, fps) của video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return total, fps


def read_frames(video_path: str, indices) -> dict:
    """Đọc tuần tự video và giữ lại các frame có chỉ số trong `indices`.

    Trả về dict {chỉ số: ảnh BGR}. Đọc tuần tự chậm hơn seek nhưng chính xác
    với mọi codec.
    """
    wanted = set(int(i) for i in indices)
    out = {}
    cap = cv2.VideoCapture(video_path)
    i = 0
    last = max(wanted) if wanted else -1
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = frame
        i += 1
    cap.release()
    return out
