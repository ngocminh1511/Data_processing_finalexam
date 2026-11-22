"""Chấm điểm độ liên quan giữa câu hỏi và từng frame.

Thay cho LAVIS/SeViLA của repo gốc bằng bản trên `transformers`, nên không cần
cài thêm thư viện nặng.
  - clip : cosine similarity (nhanh, giới hạn 77 token)
  - blip : xác suất "khớp" của đầu image-text matching (chậm hơn, thường tốt hơn)
"""
import re

import cv2
import torch
from PIL import Image

from config import SCORER_MODELS as MODELS


def question_only(q: str) -> str:
    """Bỏ phần lựa chọn A-E, chỉ giữ thân câu hỏi để chấm điểm.

    Dataset có dạng "Question: ...\\nChoices:\\nA. ..." nên bỏ thêm nhãn
    "Question:" ở đầu và "Choices:/Choice:/choose:" ở cuối (tốn token CLIP).
    """
    m = re.search(r"(\n|\s)\(?A[\.\):]\s", q)
    q = q[: m.start()] if m else q
    q = re.sub(r"^\s*Question\s*:\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*(Choices?|choose)\s*:\s*$", "", q.strip(), flags=re.IGNORECASE)
    return q.strip()


def _emb(x):
    """transformers 5.x trả về BaseModelOutputWithPooling thay vì tensor."""
    return x if torch.is_tensor(x) else x.pooler_output


class Scorer:
    def __init__(self, kind="clip", device=None, batch_size=16):
        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bs = batch_size
        name = MODELS[kind]
        if kind == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(name)
            self.proc = CLIPProcessor.from_pretrained(name)
        elif kind == "blip":
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            self.model = BlipForImageTextRetrieval.from_pretrained(name)
            self.proc = BlipProcessor.from_pretrained(name)
        else:
            raise ValueError(f"scorer phải là {list(MODELS)}")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, text: str, frames_bgr: list) -> list[float]:
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        out = []
        if self.kind == "clip":
            t = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            tf = _emb(self.model.get_text_features(**t))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            for i in range(0, len(images), self.bs):
                im = self.proc(images=images[i:i + self.bs], return_tensors="pt").to(self.device)
                f = _emb(self.model.get_image_features(**im))
                f = f / f.norm(dim=-1, keepdim=True)
                out += (f @ tf.T).squeeze(-1).tolist()
        else:
            for i in range(0, len(images), self.bs):
                batch = images[i:i + self.bs]
                inp = self.proc(images=batch, text=[text] * len(batch), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
                logits = self.model(**inp, use_itm_head=True).itm_score
                out += torch.softmax(logits, dim=1)[:, 1].tolist()
        return out
