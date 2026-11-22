"""Tập con câu hỏi cố định, dùng chung cho mọi phương pháp (giao thức mục 2.1).

    python subset.py                                  # tạo subset_1000.txt + dev_300.txt
    python subset.py --n 1000 --dev 300 --seed 0

Cần outputs/data/MCQ.parquet trên máy local:
    modal volume get urbanvideo-vol data/UrbanVideo-Bench/MCQ.parquet outputs/data/

- subset_1000: lấy mẫu phân tầng theo question_category, giữ nguyên tỉ lệ 16 task.
- dev_300: phân tầng, KHÔNG trùng subset_1000, chỉ dùng để dò siêu tham số.
Mỗi file: một Question_id trên một dòng.
"""
import argparse
import os

SUBSET_DIR = "outputs/subsets"


def load_qids(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def filter_mcq(df, limit=0, qids=None):
    """Lọc MCQ theo danh sách Question_id (ưu tiên) hoặc lấy `limit` câu đầu (chỉ để chạy thử)."""
    if qids:
        keep = set(map(str, qids))
        return df[df["Question_id"].astype(str).isin(keep)]
    return df.head(limit) if limit else df


def run_name(qids_path: str = "", limit: int = 0) -> str:
    """Tên lượt chạy theo tập câu: subset_1000, dev_300, head50, full."""
    if qids_path:
        return os.path.splitext(os.path.basename(qids_path))[0]
    return f"head{limit}" if limit else "full"


def shard_rows(df, shard, n_shards):
    """Chia theo VIDEO (không theo câu): mọi câu của một video nằm cùng shard, nên mỗi video chỉ
    được đọc/tính feature ở một container. Không đổi kết quả chọn frame."""
    videos = sorted(df["video_id"].astype(str).unique())
    mine = set(videos[shard::n_shards])
    return df[df["video_id"].astype(str).isin(mine)].to_dict("records")


def stratified_sample(df, n, seed, by="question_category"):
    """Phân bổ n câu theo tỉ lệ từng nhóm (làm tròn theo phần dư lớn nhất), rồi lấy ngẫu nhiên."""
    import pandas as pd
    counts = df[by].value_counts()
    quota = counts / counts.sum() * n
    alloc = quota.astype(int)
    rest = n - alloc.sum()
    alloc[(quota - alloc).sort_values(ascending=False).index[:rest]] += 1
    parts = [df[df[by] == c].sample(int(k), random_state=seed) for c, k in alloc.items() if k]
    return pd.concat(parts)


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq", default="outputs/data/MCQ.parquet")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.mcq)
    df["Question_id"] = df["Question_id"].astype(str)
    test = stratified_sample(df, a.n, a.seed)
    dev = stratified_sample(df[~df["Question_id"].isin(test["Question_id"])], a.dev, a.seed)

    os.makedirs(SUBSET_DIR, exist_ok=True)
    for name, part in ((f"subset_{a.n}", test), (f"dev_{a.dev}", dev)):
        path = f"{SUBSET_DIR}/{name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(sorted(part["Question_id"], key=int)) + "\n")
        print(f"{path}: {len(part)} câu, {part['video_id'].nunique()} video, "
              f"{part['question_category'].nunique()} task")
    print(test["question_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
