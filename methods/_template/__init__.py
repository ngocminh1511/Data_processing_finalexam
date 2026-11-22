"""Mẫu cho phương pháp mới (KTV, MoPrune, VETO, ...).

Cách dùng:
  1. Copy thư mục này: methods/_template -> methods/ktv
  2. Đổi tên class, cài đặt select()
  3. Thêm vào REGISTRY trong methods/__init__.py:
         "ktv": ("methods.ktv", "KTVSelector"),
  4. Chạy thử: modal run select_frames.py --method ktv --limit 50
"""
from methods.base import FrameSelector, read_frames, video_info


class TemplateSelector(FrameSelector):
    name = "template"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # nạp mô hình / tham số riêng của phương pháp ở đây

    def select(self, video_path, question, num_frames, qid=None):
        total, fps = video_info(video_path)
        # ... logic chọn frame ...
        # Có thể dùng read_frames(video_path, indices) để lấy ảnh nếu cần chấm điểm.
        # Bắt buộc: trả về list chỉ số frame, sắp xếp tăng dần, tối đa num_frames phần tử.
        raise NotImplementedError
