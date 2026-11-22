"""BƯỚC 3 — Chấm điểm và so sánh các phương pháp (chạy trên máy bạn, không cần GPU).

    python evaluate.py                          # chấm mọi file trong outputs/results/
    python evaluate.py outputs/results/aks-clip_32f__Qwen2.5-VL-7B-Instruct__subset_1000.csv

Muốn lấy kết quả trực tiếp từ Volume:
    modal volume get urbanvideo-vol results outputs/

Theo giao thức docs/giao-thuc-danh-gia-urbanvideo.md (mục 2.2, 6.3):
  - Overall (micro) kèm CI bootstrap 95%, Macro (TB 16 task), 4 nhóm, 3 nguồn video, từng task
  - unparsed, số frame TB, thời gian chọn frame (s/câu, bỏ câu lấy từ cache), suy luận (s/câu)
  - So sánh từng cặp (cùng model + cùng tập câu): McNemar exact + Holm, CI hiệu số paired bootstrap
Lưu: outputs/summary.csv, outputs/pairwise.csv
"""
import glob
import itertools
import json
import os
import re
import sys
from math import comb

import numpy as np
import pandas as pd

from config import TASK_GROUPS

TASKS = [t for ts in TASK_GROUPS.values() for t in ts]
GROUP_OF = {t: g for g, ts in TASK_GROUPS.items() for t in ts}
N_BOOT, SEED = 1000, 0


def extract_option(text) -> str | None:
    """Lấy đáp án từ câu trả lời của model (giống ý tưởng eval.py gốc, nới lỏng hơn).

    Nhận A-H: phần lớn câu có A-E nhưng Action Generation có tới G (1 câu tới H).
    """
    if not isinstance(text, str):
        return None
    m = re.search(r"Option\s*:?\s*\[?\s*\(?([A-H])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for pat in (r"^\s*\(?([A-H])[\.\)\]:]", r"\b([A-H])\b"):  # chỉ chữ in hoa
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return m.group(1)
    return None


def video_source(video_id: str) -> str:
    """AerialVLN / EmbodiedCity / RealWorld, lấy từ tiền tố của video_id."""
    return str(video_id).split("_")[0]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Question_id"] = df["Question_id"].astype(str)
    df["pred"] = df["Output"].map(extract_option)
    df["gt"] = df["answer"].map(extract_option)
    df["correct"] = (df["pred"] == df["gt"]).astype(int)
    df["group"] = df["question_category"].map(GROUP_OF)
    df["source"] = df["video_id"].map(video_source)
    if "subset" not in df:  # file cũ (trước khi có --qids) đều chạy bằng --limit
        df["subset"] = f"head{len(df)}"
    df["model"] = df["model"].str.split("/").str[-1]
    return df


def bootstrap_ci(x: np.ndarray, seed=SEED):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]) * 100


def selection_time(tag: str):
    """TB giây/câu chọn frame, bỏ các câu lấy điểm từ cache (cache_hit = True)."""
    path = f"outputs/selections/{tag}.timing.json"
    if not os.path.exists(path):
        return np.nan
    with open(path) as f:
        t = [v["s"] for v in json.load(f).values() if v.get("cache_hit") is not True]
    return float(np.mean(t)) if t else np.nan


def summarize(df: pd.DataFrame) -> dict:
    acc = lambda s: s.mean() * 100  # noqa: E731
    lo, hi = bootstrap_ci(df["correct"].to_numpy())
    by_task = df.groupby("question_category")["correct"].mean() * 100
    row = {"tag": df["tag"].iloc[0], "model": df["model"].iloc[0], "subset": df["subset"].iloc[0],
           "n": len(df), "unparsed": int(df["pred"].isna().sum()),
           "frames": df["n_frames"].mean() if "n_frames" in df else np.nan,
           "Overall": acc(df["correct"]), "CI_lo": lo, "CI_hi": hi,
           "Macro": by_task.mean()}
    row.update(df.groupby("group")["correct"].mean().mul(100).reindex(TASK_GROUPS).to_dict())
    row.update(df.groupby("source")["correct"].mean().mul(100).add_prefix("src:").to_dict())
    row["select_s"] = selection_time(row["tag"])
    row["infer_s"] = df["infer_s"].mean() if "infer_s" in df else np.nan
    row.update(by_task.reindex([t for t in TASKS if t in by_task.index]).to_dict())
    return row


def mcnemar_exact(b: int, c: int) -> float:
    """p hai phía của McNemar exact: b, c = số câu chỉ A đúng / chỉ B đúng."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni, trả về p đã hiệu chỉnh theo thứ tự đầu vào."""
    order = np.argsort(pvals)
    adj, run = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def pairwise(frames: dict) -> pd.DataFrame:
    """So sánh mọi cặp tag cùng model và cùng tập câu, trên các câu chung.

    frames: {(model, subset, tag): df}"""
    rows = []
    keyed = {}
    for model, subset, tag in frames:
        nf = re.search(r"_(\d+)f$", tag)  # chỉ so cùng số frame (8f với 8f, 32f với 32f)
        keyed.setdefault((model, subset, nf.group(1) if nf else ""), []).append(tag)
    for (model, subset, _), tags in keyed.items():
        block = []
        for a, b in itertools.combinations(sorted(tags), 2):
            m = frames[(model, subset, a)][["Question_id", "correct"]].merge(
                frames[(model, subset, b)][["Question_id", "correct"]], on="Question_id",
                suffixes=("_a", "_b"))
            xa, xb = m["correct_a"].to_numpy(), m["correct_b"].to_numpy()
            d = xa - xb
            rng = np.random.default_rng(SEED)
            boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1) * 100
            block.append({"model": model, "subset": subset, "A": a, "B": b, "n": len(m),
                          "acc_A": xa.mean() * 100, "acc_B": xb.mean() * 100,
                          "diff": d.mean() * 100,
                          "diff_lo": np.percentile(boot, 2.5), "diff_hi": np.percentile(boot, 97.5),
                          "only_A": int((d == 1).sum()), "only_B": int((d == -1).sum()),
                          "p": mcnemar_exact(int((d == 1).sum()), int((d == -1).sum()))})
        if block:
            for r, p in zip(block, holm([r["p"] for r in block])):
                r["p_holm"] = p
            rows += block
    return pd.DataFrame(rows)


def main():
    files = sys.argv[1:] or sorted(glob.glob("outputs/results/*.csv"))
    if not files:
        print("Không có file kết quả. Chạy answer.py trước.")
        return
    frames = {}  # (model, subset, tag) -> df
    for f in files:
        df = load(f)
        frames[(df["model"].iloc[0], df["subset"].iloc[0], df["tag"].iloc[0])] = df

    summary = pd.DataFrame([summarize(df) for df in frames.values()])
    summary = summary.sort_values(["model", "subset", "Overall"], ascending=[True, True, False])
    summary.to_csv("outputs/summary.csv", index=False)

    pw = pairwise(frames)
    if len(pw):
        pw.to_csv("outputs/pairwise.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 250,
                           "display.float_format", "{:.1f}".format):
        main_cols = ["tag", "model", "subset", "n", "unparsed", "frames", "Overall", "CI_lo", "CI_hi",
                     "Macro", *TASK_GROUPS, "select_s", "infer_s"]
        print("== Tổng hợp (Overall = micro, kèm CI 95% bootstrap) ==")
        out = summary[main_cols].copy()
        out[["select_s", "infer_s"]] = out[["select_s", "infer_s"]].map(lambda v: f"{v:.3f}")
        print(out.to_string(index=False))
        print("\n== Theo nguồn video ==")
        print(summary[["tag", "subset", *[c for c in summary if c.startswith("src:")]]]
              .to_string(index=False))
        print("\n== Theo task ==")
        print(summary[["tag", "subset", *[t for t in TASKS if t in summary]]].to_string(index=False))
        if len(pw):
            print("\n== So sánh từng cặp (McNemar exact, Holm trong cùng model + tập câu) ==")
            show = pw.copy()
            show["p"] = show["p"].map("{:.3g}".format)
            show["p_holm"] = show["p_holm"].map("{:.3g}".format)
            print(show.to_string(index=False))
    print("\nĐã lưu outputs/summary.csv" + (", outputs/pairwise.csv" if len(pw) else ""))


if __name__ == "__main__":
    main()
