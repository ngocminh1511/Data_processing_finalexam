"""MDP3 — chọn frame theo danh sách (list-wise), không cần huấn luyện (Sun et al., ICCV 2025).

Port từ code chính thức github.com/sunh-23/MDP3 (vlmeval/smp/mdp3_frame_selector.py, Apache-2.0):
  - Đặc trưng: SigLIP so400m-patch14-384 cho cả frame và câu hỏi (cùng không gian). Câu hỏi dài
    hơn 64 token được cắt thành nhiều đoạn rồi lấy trung bình embedding (như code gốc).
  - Kernel: tổng 5 kernel Gaussian, alpha = 2^-3..2^1; độ liên quan r_i = k(câu hỏi, frame_i).
  - Mục tiêu: (1/lambda) * 2 * sum log r_i + log det(L_S), lambda = 0.2 (DPP có điều kiện câu hỏi).
  - Tuần tự: chia ứng viên thành đoạn 32 frame; quy hoạch động qua các đoạn để phân bổ số frame,
    mỗi đoạn điều kiện trên 1 frame đã chọn gần nhất của đoạn trước; trong đoạn chọn tham lam
    (MAP) với cập nhật Cholesky.
Khác code gốc (ghi trong docs/giao-thuc-danh-gia-urbanvideo.md):
  1. Ứng viên = candidate_frames() (1 fps, tối thiểu 2K) dùng chung với AKS/KTV, lấy đều xuống
     tối đa 128 (code gốc: đúng 128 frame lấy đều).
  2. Đoạn cuối có thể ngắn hơn 32 (code gốc giả định số ứng viên chia hết cho 32); bỏ qua trạng
     thái quy hoạch động không đạt được.
  3. Không dùng clear_prompt() (chỉ dành cho prompt của các benchmark trong paper); đưa câu hỏi
     kèm các lựa chọn, giống code gốc sau khi bỏ phần hướng dẫn.
Cache: embedding frame theo video, embedding câu hỏi theo Question_id.
"""
import os

import numpy as np

from methods.base import FrameSelector, candidate_frames

SIGLIP = "google/siglip-so400m-patch14-384"
MAX_CAND = 128
NEG_INF = float(0x7fffffff)


def _kernel(X, Y=None, alphas=(2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1)):
    import torch
    Y = X if Y is None else Y
    d2 = ((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(2)
    return sum(torch.exp(-d2 / (2 * a)) for a in alphas)


def _chol_update(L, v):
    """Thêm 1 hàng/cột vào phân rã Cholesky L; trả (L mới, log det)."""
    import torch
    n = L.shape[0]
    v = v.view(-1, 1)
    vp = torch.linalg.solve_triangular(L, v[:n], upper=False)
    diag = torch.sqrt(torch.abs(v[-1] - vp.T @ vp))
    new_L = torch.zeros((n + 1, n + 1), dtype=L.dtype, device=L.device)
    new_L[:n, :n] = L
    new_L[n, :n] = vp.flatten()
    new_L[n, n] = diag
    return new_L, 2 * torch.log(torch.diag(new_L)).sum()


def _greedy_segment(total, offset, max_sel, lamda):
    """Bản vector hóa của seqdpp_select_super_fast: mỗi bước tính độ tăng log det của MỌI ứng viên
    bằng một lần giải tam giác (log det(S_{A+i}) = log det(S_A) + log|S_ii - s_i^T S_A^-1 s_i|),
    thay vì lặp từng ứng viên. Cùng mục tiêu, cùng cách phá hòa (lấy chỉ số nhỏ nhất) -> cùng
    kết quả với bản gốc (_greedy_segment_ref), nhanh hơn nhiều lần."""
    import torch
    if max_sel == 0:
        return [0.0], [[]]
    r, S = total[0, 1:], total[1:, 1:]
    n_cand = len(S) - offset
    cidx = torch.arange(offset, len(S))
    logr = torch.log(r)
    sel = list(range(offset))
    diagS = torch.diagonal(S)[cidx]
    avail = torch.ones(n_cand, dtype=torch.bool)
    trace, scores = [], [0.0]
    while len(trace) < max_sel:
        if sel:
            A = torch.tensor(sel)
            L = torch.linalg.cholesky(S[A][:, A])
            logdet_A = 2 * torch.log(torch.diagonal(L)).sum()
            X = torch.linalg.solve_triangular(L, S[A][:, cidx], upper=False)
            d2 = diagS - (X ** 2).sum(0)
            base = logr[A].sum()
        else:
            logdet_A, d2, base = 0.0, diagS.clone(), 0.0
        obj = 1.0 / lamda * 2 * (base + logr[cidx]) + logdet_A + torch.log(torch.abs(d2))
        obj = torch.where(avail, obj, torch.full_like(obj, -float("inf")))
        j = int(torch.argmax(obj))  # argmax trả chỉ số nhỏ nhất khi hòa
        scores.append(float(obj[j]))
        trace.append(j)
        avail[j] = False
        sel.append(j + offset)
    return scores, [sorted(trace[:k]) for k in range(len(trace) + 1)]


def _greedy_segment_ref(total, offset, max_sel, lamda):
    """seqdpp_select_super_fast của code gốc (giữ để kiểm tra tương đương)."""
    import torch
    if max_sel == 0:
        return [0.0], [[]]
    r, S = total[0:1, 1:], total[1:, 1:]
    cand = list(range(len(S) - offset))
    cond = list(range(offset))
    L = torch.linalg.cholesky(S[cond][:, cond]) if cond else None
    trace, scores = [], [0.0]
    while len(trace) < max_sel:
        best, best_i, best_L = -NEG_INF, -1, None
        for i in cand:
            if i in trace:
                continue
            ci = i + offset
            sel = cond + [j + offset for j in trace] + [ci]
            if L is None:
                v = S[sel][:, sel]
                cur_L, logdet = torch.sqrt(v).reshape(1, 1), v.clone().log().reshape(())
            else:
                cur_L, logdet = _chol_update(L, S[ci:ci + 1][:, sel])
            obj = 1.0 / lamda * 2 * torch.log(r[:, sel]).sum() + logdet
            if best_i == -1 or obj > best:
                best, best_i, best_L = obj, i, cur_L
        scores.append(float(best))
        trace.append(best_i)
        L = best_L
    return scores, [sorted(trace[:j]) for j in range(len(trace) + 1)]


def mdp3_select(img, txt, k, lamda=0.2, segment_size=32, condition_size=1):
    """img: (n, d) embedding frame; txt: (1, d) embedding câu hỏi. Trả chỉ số (trong 0..n-1)."""
    import torch
    n = len(img)
    if n <= k:
        return list(range(n))
    seg_num = (n + segment_size - 1) // segment_size
    dp = [[0.0] + [-NEG_INF] * k for _ in range(seg_num + 1)]
    tr = [[[] for _ in range(k + 1)] for _ in range(seg_num + 1)]
    for s in range(1, seg_num + 1):
        idx = list(range((s - 1) * segment_size, min(s * segment_size, n)))
        cand = img[idx]
        sim = _kernel(cand)
        for start in range(0, min(k, (s - 1) * segment_size) + 1):
            if dp[s - 1][start] <= -NEG_INF:  # trạng thái không đạt được
                continue
            cond_idx = tr[s - 1][start][-min(condition_size, len(tr[s - 1][start])):] \
                if tr[s - 1][start] else []
            add = torch.cat([txt.reshape(1, -1), img[cond_idx]]) if cond_idx else txt.reshape(1, -1)
            extra = _kernel(add, torch.cat([add, cand]))
            total = torch.cat([extra, torch.cat([extra[:, -len(sim):].T, sim], dim=1)], dim=0)
            max_sel = min(k - start, len(idx))
            scores, traces = _greedy_segment(total, len(cond_idx), max_sel, lamda)
            for m, (sc, t) in enumerate(zip(scores, traces)):
                val = dp[s - 1][start] + sc
                if val > dp[s][start + m]:
                    dp[s][start + m] = val
                    tr[s][start + m] = tr[s - 1][start] + [idx[0] + j for j in t]
    return sorted(tr[seg_num][k])


class MDP3Selector(FrameSelector):
    name = "mdp3"

    def __init__(self, cand_fps=1.0, lamda=0.2, segment_size=32, cand_k=None, batch=32,
                 cache_dir="cache/mdp3", **kwargs):
        super().__init__(**kwargs)
        self.cand_fps, self.lamda, self.segment_size = cand_fps, lamda, segment_size
        self.cand_k, self.batch, self.cache_dir = cand_k, batch, cache_dir
        os.makedirs(os.path.join(cache_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "text"), exist_ok=True)
        self._model = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        from transformers import SiglipModel, SiglipProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SiglipModel.from_pretrained(SIGLIP).to(self.device).eval()
        self._proc = SiglipProcessor.from_pretrained(SIGLIP)

    def _candidates(self, video_path, num_frames):
        cand = candidate_frames(video_path, self.cand_k or num_frames, self.cand_fps)
        if len(cand) > MAX_CAND:
            cand = [cand[i] for i in np.linspace(0, len(cand) - 1, MAX_CAND).round().astype(int)]
        return cand

    def _frame_embeds(self, video_path, num_frames):
        import torch
        from PIL import Image

        from methods.base import read_frames
        path = os.path.join(self.cache_dir, "frames",
                            f"{os.path.basename(video_path)}_k{self.cand_k or num_frames}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return z["emb"], z["frames"].tolist(), True
        self.warmup()
        import cv2
        cand = self._candidates(video_path, num_frames)
        frames = read_frames(video_path, cand)
        kept = [i for i in cand if i in frames]
        embs = []
        with torch.no_grad():
            for s in range(0, len(kept), self.batch):
                ims = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                       for i in kept[s:s + self.batch]]
                px = self._proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
                e = self._model.get_image_features(pixel_values=px)
                e = getattr(e, "pooler_output", e)
                embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        emb = np.concatenate(embs)
        np.savez(path, emb=emb, frames=np.asarray(kept))
        return emb, kept, False

    def _text_embed(self, question, qid):
        import torch
        path = os.path.join(self.cache_dir, "text", f"{qid}.npy") if qid is not None else None
        if path and os.path.exists(path):
            return np.load(path)
        self.warmup()
        ids = self._proc(text=[question], padding="max_length", return_tensors="pt")["input_ids"]
        # cắt thành các đoạn <= 64 token (lấy từ hai đầu như code gốc) rồi lấy trung bình
        n = ids.shape[-1]
        stride_num = (n + 63) // 64
        stride = (n + stride_num - 1) // stride_num
        heads, tails, lo, hi = [], [], 0, n
        while lo < hi:
            heads.append(ids[:, lo:lo + stride])
            lo += stride
            if lo < hi:
                tails.append(ids[:, hi - stride:hi])
                hi -= stride
        chunks = [c for c in heads + tails[::-1]]
        width = max(c.shape[-1] for c in chunks)
        pad = self._proc.tokenizer.pad_token_id or 0
        chunks = [torch.nn.functional.pad(c, (0, width - c.shape[-1]), value=pad) for c in chunks]
        with torch.no_grad():
            e = self._model.get_text_features(input_ids=torch.cat(chunks).to(self.device))
            e = getattr(e, "pooler_output", e)
            e = torch.nn.functional.normalize(e.float(), dim=-1).mean(0, keepdim=True)
        e = e.cpu().numpy()
        if path:
            np.save(path, e)
        return e

    def select(self, video_path, question, num_frames, qid=None):
        import torch
        emb, cand, hit_v = self._frame_embeds(video_path, num_frames)
        txt = self._text_embed(question, qid)
        self.last_cache_hit = hit_v and qid is not None and os.path.exists(
            os.path.join(self.cache_dir, "text", f"{qid}.npy"))
        if len(cand) <= num_frames:
            return sorted(int(c) for c in cand)
        pick = mdp3_select(torch.from_numpy(emb).double(), torch.from_numpy(txt).double(),
                           num_frames, self.lamda, self.segment_size)
        return sorted(int(cand[i]) for i in pick)
