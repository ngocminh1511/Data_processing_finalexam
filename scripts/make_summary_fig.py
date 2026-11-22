"""Ảnh tổng hợp kết quả (đọc outputs/final_table.csv, chạy sau scripts/final_table.py).

(1) Overall 32 frame theo mô hình (InternVL2-8B, Qwen2.5-VL-7B), kèm mức 0 frame để tham chiếu
(2) Hiệu số so với Uniform-32 kèm CI 95% (paired bootstrap) và p McNemar
(3) Bảng mọi cấu hình, xếp theo mô hình (LLaVA-NeXT-Video là kết quả phụ)
(4) Nhận xét

    python scripts/make_summary_fig.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
T = pd.read_csv("outputs/final_table.csv")

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METHODS = ["Uniform", "AKS", "KTV", "MDP3"]
MCOL = {"Uniform": "#b9b7b0", "AKS": "#9cc3ee", "KTV": "#4a8fd9", "MDP3": "#1b4f8f"}
MODELS = ["InternVL2-8B", "Qwen2.5-VL-7B"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2})


def vn(v, k=1, sign=False):
    return "—" if pd.isna(v) else (f"{v:+.{k}f}" if sign else f"{v:.{k}f}").replace(".", ",")


def pfmt(v):
    return "" if pd.isna(v) else ("p < 0,001" if v < 0.001 else f"p = {v:.3f}".replace(".", ","))


def base(meth):
    return meth.replace("+E", "")


def row(m, meth, nf):
    s = T[(T["Mô hình"] == m) & (T["Frame"] == nf) & (T["Phương pháp"].map(base) == meth)]
    return s.iloc[0]


fig = plt.figure(figsize=(17, 13.2), facecolor=SURF)
fig.text(0.03, 0.968, "Chọn keyframe cho Video-LLM trên UrbanVideo-Bench — tổng hợp kết quả",
         fontsize=19, weight="bold")
fig.text(0.03, 0.947, "subset_1000 (1.000 câu phân tầng theo 16 task) · greedy decoding · H100, vLLM 0.31 · "
         "Overall = tỉ lệ đúng (micro) · CI 95% bootstrap · p = McNemar exact ghép cặp",
         fontsize=10.5, color=INK2)

# ---------- (1) Overall 32 frame theo mô hình ----------
ax = fig.add_axes([0.05, 0.635, 0.45, 0.27], facecolor=SURF)
w = 0.19
for gi, m in enumerate(MODELS):
    zero = row(m, "0 frame", 0)["Overall"]
    ax.plot([gi - 2.3 * w, gi + 2.3 * w], [zero, zero], ls="--", lw=1.3, color=MUTED, zorder=2)
    ax.text(gi + 2.35 * w, zero, f"0 frame\n{vn(zero)}", va="center", ha="left", fontsize=8.5, color=INK2)
    for k, meth in enumerate(METHODS):
        r = row(m, meth, 32)
        lo, hi = map(float, r["CI95"].split("–"))
        xk = gi + (k - 1.5) * w
        ax.bar(xk, r["Overall"], width=w * 0.92, color=MCOL[meth], zorder=3)
        ax.errorbar(xk, r["Overall"], yerr=[[r["Overall"] - lo], [hi - r["Overall"]]], fmt="none",
                    ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
        star = m == "InternVL2-8B" and meth == "MDP3"
        ax.text(xk, hi + 0.3, vn(r["Overall"]), ha="center", va="bottom", fontsize=10,
                weight="bold" if star else "normal")
ax.set_xticks(range(len(MODELS)), MODELS, fontsize=11)
ax.set_xlim(-0.55, len(MODELS) - 0.25)
ax.set_ylim(26, 41)
ax.set_ylabel("Overall (%)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Overall (%) ở 32 frame, CI 95% — nét đứt: cùng mô hình khi không có video",
             loc="left", fontsize=12, weight="bold", pad=10)
labels = {"Uniform": "Uniform", "AKS": "AKS(+E)", "KTV": "KTV+E", "MDP3": "MDP3+E"}
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MCOL[k]) for k in METHODS],
          labels=[labels[k] for k in METHODS], ncol=4, frameon=False, loc="upper left", fontsize=9.5)

# ---------- (2) Hiệu số so với Uniform-32 ----------
ax2 = fig.add_axes([0.62, 0.635, 0.235, 0.27], facecolor=SURF)
items = [(m, meth) for m in MODELS for meth in ("AKS", "KTV", "MDP3")]
ys = np.array([7, 6, 5, 2.6, 1.6, 0.6])
for y, (m, meth) in zip(ys, items):
    r = row(m, meth, 32)
    sig = r["p vs Uniform"] < 0.05
    ax2.plot([r["vsU_lo"], r["vsU_hi"]], [y, y], color=MCOL[meth], lw=2.4, zorder=3)
    ax2.plot(r["vs Uniform"], y, "o", ms=8, color=MCOL[meth], mec=INK if sig else MCOL[meth], zorder=4)
    ax2.text(3.2, y, f"{vn(r['vs Uniform'], sign=True)}  ({pfmt(r['p vs Uniform'])})", va="center",
             fontsize=9.5, weight="bold" if sig else "normal", color=INK)
ax2.axvline(0, color=INK2, lw=1)
ax2.set_yticks(ys, [f"{labels[meth]}" for _, meth in items])
for y, m in ((7.75, MODELS[0]), (3.35, MODELS[1])):
    ax2.text(-3.0, y, m, fontsize=10.5, weight="bold", va="center")
ax2.set_xlim(-3, 3)
ax2.set_ylim(0, 8.2)
ax2.set_xlabel("điểm hơn Uniform-32 (CI 95% paired bootstrap)")
ax2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("Hơn Uniform bao nhiêu? (cùng mô hình, cùng 1.000 câu)", loc="left", fontsize=12,
              weight="bold", pad=10)

# ---------- (3) Bảng ----------
axt = fig.add_axes([0.03, 0.275, 0.94, 0.31])
axt.axis("off")
axt.set_title("Bảng kết quả đầy đủ (subset_1000), xếp theo mô hình", loc="left", fontsize=12,
              weight="bold", pad=4)
hdr = ["Mô hình", "Frame", "Phương pháp", "Overall", "CI 95%", "Recall", "Perc.", "Reas.", "Nav.",
       "so Uniform [CI 95%] (p)", "so 0 frame (p)", "Chọn frame", "Suy luận s/câu"]
cells, meta, prev = [], [], None
for _, r in T.iterrows():
    name = r["Mô hình"] + (" (phụ)" if r["Vai trò"] == "phụ" else "")
    vu = "—" if pd.isna(r["vs Uniform"]) else \
        f"{vn(r['vs Uniform'], sign=True)} [{vn(r['vsU_lo'], sign=True)}; {vn(r['vsU_hi'], sign=True)}] " \
        f"({'<0,001' if r['p vs Uniform'] < 0.001 else vn(r['p vs Uniform'], 3)})"
    v0 = "—" if pd.isna(r["vs 0 frame"]) else \
        f"{vn(r['vs 0 frame'], sign=True)} ({'<0,001' if r['p vs 0f'] < 0.001 else vn(r['p vs 0f'], 3)})"
    cells.append([name if name != prev else "", str(r["Frame"]), r["Phương pháp"], vn(r["Overall"]),
                  r["CI95"].replace(".", ","), vn(r["Recall"]), vn(r["Perception"]), vn(r["Reasoning"]),
                  vn(r["Navigation"]), vu, v0, r["Chọn frame"], vn(r["Suy luận (s/câu)"], 3)])
    meta.append((r["Mô hình"], r["Phương pháp"], r["Vai trò"], name != prev,
                 r["Mô hình"] == "InternVL2-8B" and r["Phương pháp"] == "MDP3+E"))
    prev = name
tb = axt.table(cellText=cells, colLabels=hdr, loc="upper center", cellLoc="center",
               colWidths=[0.135, 0.04, 0.075, 0.05, 0.065, 0.045, 0.045, 0.045, 0.045, 0.15, 0.085,
                          0.13, 0.06])
tb.auto_set_font_size(False)
tb.set_fontsize(8.7)
tb.scale(1, 1.3)
for (ri, ci), cell in tb.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.6)
    if ri == 0:
        cell.set_facecolor("#efeee9")
        cell.set_text_props(weight="bold")
        continue
    m, meth, role, first, star = meta[ri - 1]
    cell.set_facecolor("#f3f2ee" if meth == "0 frame" else SURF)
    if star:
        cell.set_facecolor("#e3eefb")
        cell.set_text_props(weight="bold")
    if role == "phụ":
        cell.set_text_props(color=MUTED)
    if first and ri > 1:
        cell.visible_edges = "closed"
        cell.set_linewidth(0.6)
    if ci == 0:
        cell.set_text_props(ha="left", weight="bold", color=MUTED if role == "phụ" else INK)
        cell._loc = "left"

# ---------- (4) Nhận xét ----------
axn = fig.add_axes([0.03, 0.015, 0.94, 0.235])
axn.axis("off")
axn.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.015",
                             transform=axn.transAxes, facecolor="#f5f8fc", edgecolor="#c9daf0", lw=1))
axn.text(0.012, 0.93, "Nhận xét", fontsize=12.5, weight="bold", transform=axn.transAxes, va="top")
iv = {k: row("InternVL2-8B", k, 32) for k in METHODS}
notes = [
    f"1.  InternVL2-8B (32 frame): MDP3+E cao nhất ({vn(iv['MDP3']['Overall'])}) và là phương pháp DUY NHẤT hơn Uniform có ý nghĩa thống kê "
    f"(+1,5 điểm, CI [+0,5; +2,6], {iv['MDP3']['thắng/thua vs U']} câu, p = 0,008; Holm 0,041).",
    "     KTV+E +1,0 (p = 0,076) và AKS+E −0,1 chưa khác Uniform; MDP3+E hơn KTV+E chỉ +0,5 (p = 0,42).",
    "2.  Qwen2.5-VL-7B (32 frame): không phương pháp nào khác Uniform (−0,7 … +0,2 điểm, mọi p > 0,4); khoảng 90% câu trả lời "
    "không đổi khi đổi phương pháp → kết quả của MDP3 trên InternVL2 không lặp lại trên Qwen.",
    "3.  Video đóng góp ít: không có video vẫn đạt 30,8 (InternVL2) và 30,1 (Qwen), ngẫu nhiên = 19,7; video chỉ thêm +3,5 (p = 0,009) "
    "và +2,2 (p = 0,088); 68–73% câu trả lời giống hệt khi bỏ video",
    "     → lợi ích của chọn frame bị giới hạn bởi mức mô hình dựa vào video: có ở InternVL2 (dựa nhiều hơn), không có ở Qwen.",
    "4.  InternVL2 ở 8 frame ≈ 32 frame (Uniform 34,1 so với 34,3) nhưng suy luận nhanh hơn ~3,5 lần; ở 8 frame các phương pháp như nhau.",
    "5.  Giới hạn: subset 1.000 câu (CI ≈ ±3 điểm); chưa tái hiện 28,1 của paper (0 frame của pipeline đã 30,8 → khác thiết lập); "
    "Qwen Uniform/AKS chạy 10/2026 (AKS bản đầu, t1 = 0,8, không +E),",
    "     KTV+E/MDP3+E chạy 11/2026, cùng H100 batch 32. LLaVA-NeXT-Video (phụ): ngữ cảnh 4.096 token nên chỉ 8 frame, 58,5% câu đúng "
    "định dạng, chọn A 61% → không dùng để so sánh mô hình.",
    "     Chi phí chọn frame (T4, lần tính đầu): Uniform 0,05 s/câu < AKS 1,56 s/câu < KTV 7,6 s/video < MDP3 18,6 s/câu.",
]
for i, t in enumerate(notes):
    axn.text(0.015, 0.8 - i * 0.088, t, fontsize=9.7, transform=axn.transAxes, va="top")

os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/tong_hop_ket_qua.png", dpi=150, facecolor=SURF)
print("saved outputs/figures/tong_hop_ket_qua.png")
