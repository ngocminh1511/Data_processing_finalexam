"""Bảng tổng hợp mọi lượt chạy trên subset_1000 -> outputs/final_table.csv (+ in bảng markdown).

Thứ tự: theo mô hình (InternVL2-8B, Qwen2.5-VL-7B, LLaVA-NeXT-Video-7B), trong mỗi mô hình:
0 frame (tham chiếu) -> 32 frame -> 8 frame, phương pháp theo thứ tự Uniform, AKS, KTV, MDP3.
Kết quả chạy trên full (Qwen Uniform/AKS) được lọc đúng 1.000 câu của subset_1000.

    python scripts/final_table.py
"""
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from config import TASK_GROUPS
from evaluate import bootstrap_ci, load, mcnemar_exact
from subset import load_qids

ids = set(load_qids("outputs/subsets/subset_1000.txt"))
R = "outputs/results/"
IV, QW, LL = "InternVL2-8B", "Qwen2.5-VL-7B", "LLaVA-NeXT-Video-7B"
QWF, LLF = "Qwen2.5-VL-7B-Instruct", "LLaVA-NeXT-Video-7B-hf"
# (mô hình, phương pháp, số frame, file, vai trò)  vai trò: chính / phụ
runs = [
    (IV, "0 frame", 0, f"textonly_0f__{IV}__subset_1000.csv", "chính"),
    (IV, "Uniform", 32, f"uniform_32f__{IV}__subset_1000.csv", "chính"),
    (IV, "AKS+E", 32, f"aks-clip-t0.2-e_32f__{IV}__subset_1000.csv", "chính"),
    (IV, "KTV+E", 32, f"ktv-e_32f__{IV}__subset_1000.csv", "chính"),
    (IV, "MDP3+E", 32, f"mdp3-e_32f__{IV}__subset_1000.csv", "chính"),
    (IV, "Uniform", 8, f"uniform_8f__{IV}__subset_1000.csv", "chính"),
    (IV, "AKS+E", 8, f"aks-clip-t0.3-e_8f__{IV}__subset_1000.csv", "chính"),
    (IV, "KTV+E", 8, f"ktv-e_8f__{IV}__subset_1000.csv", "chính"),
    (QW, "0 frame", 0, f"textonly_0f__{QWF}__subset_1000.csv", "chính"),
    (QW, "Uniform", 32, f"uniform_32f__{QWF}__full.csv", "chính"),
    (QW, "AKS", 32, f"aks-clip_32f__{QWF}__full.csv", "chính"),
    (QW, "KTV+E", 32, f"ktv-e_32f__{QWF}__subset_1000.csv", "chính"),
    (QW, "MDP3+E", 32, f"mdp3-e_32f__{QWF}__subset_1000.csv", "chính"),
    (LL, "0 frame", 0, f"textonly_0f__{LLF}__subset_1000.csv", "phụ"),
    (LL, "Uniform", 8, f"uniform_8f__{LLF}__subset_1000.csv", "phụ"),
    (LL, "AKS+E", 8, f"aks-clip-t0.3-e_8f__{LLF}__subset_1000.csv", "phụ"),
    (LL, "KTV+E", 8, f"ktv-e_8f__{LLF}__subset_1000.csv", "phụ"),
]
# t1 của AKS (dò trên dev_300 cho InternVL2; Qwen dùng bản đầu tiên, t1 mặc định, không +E)
T1 = {(IV, 32): 0.2, (IV, 8): 0.3, (QW, 32): 0.8, (LL, 8): 0.3}
SEL = {"Uniform": "0,05 s/câu (CPU)", "AKS": "1,56 s/câu (CLIP, T4)", "KTV": "7,6 s/video (DINOv2, T4)",
       "MDP3": "18,6 s/câu (SigLIP, T4)"}

D = {}
for m, meth, nf, f, _ in runs:
    d = load(R + f)
    D[(m, meth, nf)] = d[d.Question_id.isin(ids)].set_index("Question_id").sort_index()
    assert len(D[(m, meth, nf)]) == 1000, (m, meth, nf, len(D[(m, meth, nf)]))

rows = []
for m, meth, nf, f, role in runs:
    d = D[(m, meth, nf)]
    b0, u = D[(m, "0 frame", 0)], D.get((m, "Uniform", nf))
    lo, hi = bootstrap_ci(d.correct.to_numpy())
    r = {"Mô hình": m, "Vai trò": role, "Frame": nf, "Phương pháp": meth,
         "t1": T1.get((m, nf)) if meth.startswith("AKS") else np.nan, "n": len(d),
         "Overall": 100 * d.correct.mean(), "CI95": f"{lo:.1f}–{hi:.1f}",
         "Macro": 100 * d.groupby("question_category").correct.mean().mean()}
    for g in TASK_GROUPS:
        r[g] = 100 * d[d.group == g].correct.mean()
    x0 = d.correct - b0.correct
    is0 = meth == "0 frame"
    r["vs 0 frame"] = np.nan if is0 else 100 * x0.mean()
    r["p vs 0f"] = np.nan if is0 else mcnemar_exact(int((x0 == 1).sum()), int((x0 == -1).sum()))
    if u is not None and meth not in ("Uniform", "0 frame"):
        xu = (d.correct - u.correct).to_numpy()
        dlo, dhi = bootstrap_ci(xu)          # CI của hiệu số (paired bootstrap)
        r.update({"vs Uniform": 100 * xu.mean(), "vsU_lo": dlo, "vsU_hi": dhi,
                  "thắng/thua vs U": f"{int((xu == 1).sum())}/{int((xu == -1).sum())}",
                  "p vs Uniform": mcnemar_exact(int((xu == 1).sum()), int((xu == -1).sum())),
                  "% giống Uniform": 100 * (d.pred == u.pred).mean()})
    r["% giống 0f"] = np.nan if is0 else 100 * (d.pred == b0.pred).mean()
    r["% đúng định dạng"] = 100 * d.Output.astype(str).str.strip().str.startswith("Option").mean()
    r["% chọn A"] = 100 * (d.pred == "A").mean()
    r["Chọn frame"] = "—" if is0 else SEL[meth.replace("+E", "")]
    r["Suy luận (s/câu)"] = d.infer_s.mean()
    rows.append(r)

T = pd.DataFrame(rows)
T.to_csv("outputs/final_table.csv", index=False)


def f1(v):
    return "—" if pd.isna(v) else f"{v:.1f}"


def sg(v):
    return "—" if pd.isna(v) else f"{v:+.1f}"


def pv(v):
    return "—" if pd.isna(v) else (f"{v:.3f}" if v >= 0.001 else "<0,001")


print("| Mô hình | Frame | Phương pháp | Overall | CI 95% | Recall | Perc. | Reas. | Nav. | vs Uniform [CI] (p) | vs 0 frame (p) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in T.iterrows():
    vu = "—" if pd.isna(r.get("vs Uniform")) else \
        f"{sg(r['vs Uniform'])} [{r['vsU_lo']:+.1f}; {r['vsU_hi']:+.1f}] ({pv(r['p vs Uniform'])})"
    v0 = "—" if pd.isna(r["vs 0 frame"]) else f"{sg(r['vs 0 frame'])} ({pv(r['p vs 0f'])})"
    print(f"| {r['Mô hình']}{' (phụ)' if r['Vai trò'] == 'phụ' else ''} | {r['Frame']} | {r['Phương pháp']} | "
          f"**{r['Overall']:.1f}** | {r['CI95']} | {f1(r['Recall'])} | {f1(r['Perception'])} | "
          f"{f1(r['Reasoning'])} | {f1(r['Navigation'])} | {vu} | {v0} |")
