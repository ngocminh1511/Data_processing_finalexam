"""Smoke test local (không cần Modal/GPU): video giả lập + hợp đồng dữ liệu mục 6."""
import json

import cv2
import numpy as np
import pytest

from evaluate import extract_option
from methods import get_method
from methods.aks.ada import ada_select
from methods.base import read_frames, video_info


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """Video 300 frame, 30 fps; frame i có độ sáng (i % 10) * 25 để kiểm tra đúng chỉ số."""
    path = str(tmp_path_factory.mktemp("v") / "fake.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(300):
        w.write(np.full((48, 64, 3), (i % 10) * 25, np.uint8))
    w.release()
    return path


def _check_contract(idx, num_frames):
    assert idx == sorted(set(idx)) and len(idx) <= num_frames
    assert all(isinstance(i, int) for i in idx)


def test_video_info_and_read_frames(video):
    total, fps = video_info(video)
    assert total == 300 and fps == pytest.approx(30)
    frames = read_frames(video, [0, 103, 299])
    assert sorted(frames) == [0, 103, 299]
    for i, f in frames.items():
        assert abs(f.mean() - (i % 10) * 25) < 10


def test_uniform(video):
    idx = get_method("uniform", cache_dir="unused").select(video, "q", 32)
    _check_contract(idx, 32)
    assert len(idx) == 32 and idx[0] == 0 and idx[-1] == 299


def test_ada():
    rng = np.random.default_rng(0)
    scores = rng.random(300)
    idx = ada_select(scores, list(range(300)), 32)
    _check_contract(idx, 32)
    assert ada_select(scores[:10], list(range(10)), 32) == list(range(10))


def test_aks_from_cache(video, tmp_path):
    cand = list(range(0, 300, 5))
    (tmp_path / "clip").mkdir()
    (tmp_path / "clip" / "q1.json").write_text(
        json.dumps({"scores": np.linspace(0, 1, len(cand)).tolist(), "frames": cand}))
    sel = get_method("aks", scorer="clip", cache_dir=str(tmp_path))
    idx = sel.select(video, "q", 32, qid="q1")
    _check_contract(idx, 32)
    assert set(idx) <= set(cand)


@pytest.mark.parametrize("text, expected", [
    ("Option: [B]; Reason: ...", "B"),
    ("option: c; reason", "C"),
    ("(D) because ...", "D"),
    ("E", "E"),
    ("Option: [G]; Reason: I descend", "G"),
    ("F", "F"),
    ("I am not sure", None),
    (None, None),
])
def test_extract_option(text, expected):
    assert extract_option(text) == expected


def test_candidate_frames(video):
    from methods.base import candidate_frames
    # video 10 s @30fps: 1 fps chỉ có 10 ứng viên -> nâng lên 2 × 32 = 64
    c = candidate_frames(video, 32)
    assert len(c) == 64 and c[0] == 0 and c[-1] == 299
    assert len(candidate_frames(video, 200)) == 300  # không vượt tổng số frame


def test_stratified_subset():
    import pandas as pd

    from subset import filter_mcq, stratified_sample
    df = pd.DataFrame({"Question_id": range(1000),
                       "question_category": ["a"] * 700 + ["b"] * 290 + ["c"] * 10})
    s = stratified_sample(df, 100, seed=0)
    assert len(s) == 100 and s["question_category"].value_counts().to_dict() == {"a": 70, "b": 29, "c": 1}
    assert s.equals(stratified_sample(df, 100, seed=0))  # cố định theo seed
    assert len(filter_mcq(df, qids=["1", "5"])) == 2 and len(filter_mcq(df, limit=7)) == 7


def test_mcnemar_holm():
    from evaluate import holm, mcnemar_exact
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact(3, 12) == pytest.approx(mcnemar_exact(12, 3))
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


# ---- Biến thể +E, tên tag, KTV, chia shard theo video (kế hoạch InternVL2) ----

def test_add_endpoints():
    from methods.base import add_endpoints
    # đã có đầu/cuối -> không đổi
    assert add_endpoints([0, 50, 99], 100, 32) == [0, 50, 99]
    # thêm đầu/cuối, bỏ đúng 2 frame thừa nhất (khoảng cách tới frame kề nhỏ nhất)
    sel = [10, 11, 30, 50, 70, 90]
    out = add_endpoints(sel, 100, 6)
    assert out[0] == 0 and out[-1] == 99 and len(out) == 6
    assert 10 not in out or 11 not in out          # cặp sát nhau bị bỏ bớt
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = sorted(set(rng.integers(0, 1000, 32).tolist()))
        o = add_endpoints(s, 1000, 32)
        _check_contract(o, 32)
        assert o[0] == 0 and o[-1] == 999 and set(o) <= set(s) | {0, 999}
    assert add_endpoints([3, 5], 10, 32) == [0, 3, 5, 9]  # video ngắn: chỉ thêm, không bỏ


def test_selection_tag():
    from config import selection_tag
    assert selection_tag("uniform", 32) == "uniform_32f"
    assert selection_tag("aks", 32, {"scorer": "clip"}) == "aks-clip_32f"
    assert selection_tag("aks", 32, {"scorer": "clip", "t1": 0.3, "endpoints": True}) == "aks-clip-t0.3-e_32f"
    assert selection_tag("ktv", 32, {"endpoints": True}) == "ktv-e_32f"


def test_ktv_select():
    from methods.ktv import ktv_select
    rng = np.random.default_rng(0)
    # 4 cụm rõ ràng, mỗi cụm 25 frame liền nhau
    feats = np.concatenate([rng.normal(c * 10, 0.1, (25, 8)) for c in range(4)])
    ids = list(range(0, 1000, 10))
    out = ktv_select(feats, ids, 4)
    _check_contract(out, 4)
    assert len(out) == 4 and [ids.index(f) // 25 for f in out] == [0, 1, 2, 3]  # mỗi cụm 1 frame
    out32 = ktv_select(rng.normal(size=(100, 8)), ids, 32)
    _check_contract(out32, 32)
    assert len(out32) == 32 and set(out32) <= set(ids)
    assert ktv_select(feats[:5], ids[:5], 32) == ids[:5]
    assert ktv_select(feats, ids, 4) == out  # cố định theo seed


def test_ktv_from_cache(video, tmp_path):
    import os
    cand = list(range(0, 300, 3))
    np.savez(tmp_path / f"{os.path.basename(video)}_k32.npz",
             feats=np.random.default_rng(1).normal(size=(len(cand), 16)), frames=np.asarray(cand))
    sel = get_method("ktv", cache_dir=str(tmp_path))
    idx = sel.select(video, "q", 32)
    assert sel.last_cache_hit is True
    _check_contract(idx, 32)
    assert len(idx) == 32 and set(idx) <= set(cand)


def test_shard_rows_by_video():
    import pandas as pd

    from subset import run_name, shard_rows
    df = pd.DataFrame({"Question_id": range(20), "video_id": [f"v{i % 7}.mp4" for i in range(20)]})
    parts = [shard_rows(df, s, 3) for s in range(3)]
    qids = sorted(r["Question_id"] for p in parts for r in p)
    assert qids == list(range(20))                                   # đủ, không trùng
    vids = [{r["video_id"] for r in p} for p in parts]
    assert all(not (vids[i] & vids[j]) for i in range(3) for j in range(i + 1, 3))  # mỗi video 1 shard
    assert run_name("outputs/subsets/dev_300.txt") == "dev_300"
    assert run_name(limit=20) == "head20" and run_name() == "full"


def test_llm_kwargs():
    from answer import is_video_llm, llava_prompt, llm_kwargs
    assert llm_kwargs("OpenGVLab/InternVL2-8B", 32768, 32) == {
        "max_model_len": 16384, "limit_mm_per_prompt": {"image": 32},
        "mm_processor_kwargs": {"max_dynamic_patch": 1}}
    assert llm_kwargs("Qwen/Qwen2.5-VL-7B-Instruct", 32768, 32) == {
        "max_model_len": 32768, "limit_mm_per_prompt": {"image": 32}}
    assert llm_kwargs("llava-hf/LLaVA-NeXT-Video-7B-hf", 32768, 8) == {
        "max_model_len": 4096, "limit_mm_per_prompt": {"video": 1}}
    assert is_video_llm("llava-hf/LLaVA-NeXT-Video-7B-hf") and not is_video_llm("OpenGVLab/InternVL2-8B")
    assert llava_prompt("Q", True) == "USER: <video>\nQ ASSISTANT:"
    assert llava_prompt("Q", False) == "USER: Q ASSISTANT:"


# ---- Ngân sách 8 frame (stage f8) ----

def test_ada_k8_needs_depth3():
    rng = np.random.default_rng(0)
    scores, ids = rng.random(200), list(range(200))
    assert ada_select(scores, ids, 8, t1=0.8, all_depth=5) == []      # lý do phải chỉnh all_depth
    for t1 in (0.8, 0.3, 0.2):
        out = ada_select(scores, ids, 8, t1=t1, all_depth=3)
        _check_contract(out, 8)
        assert len(out) == 8


def test_pipeline_f8_params():
    import pipeline
    m = pipeline.aks(0.2, 8)
    assert m[1]["all_depth"] == 3 and "all_depth" not in pipeline.aks(0.2)[1]
    assert pipeline.tag_of(m, 8) == "aks-clip-t0.2-e_8f"
    assert pipeline.tag_of(pipeline.KTV8, 8) == "ktv-e_8f"
    assert pipeline.tag_of(pipeline.UNIFORM, 8) == "uniform_8f"


def test_ktv_cand_k_reuses_32_cache(video, tmp_path):
    import os
    cand = list(range(0, 300, 3))
    np.savez(tmp_path / f"{os.path.basename(video)}_k32.npz",
             feats=np.random.default_rng(2).normal(size=(len(cand), 16)), frames=np.asarray(cand))
    sel = get_method("ktv", cache_dir=str(tmp_path), cand_k=32)
    idx = sel.select(video, "q", 8)
    assert sel.last_cache_hit is True and len(idx) == 8 and set(idx) <= set(cand)


def test_text_only_baseline_tag():
    import pipeline
    from config import TEXT_ONLY_TAG
    assert pipeline.tag_of(pipeline.TEXT_ONLY, 0) == TEXT_ONLY_TAG == "textonly_0f"


# ---- MDP3 ----

def test_mdp3_registry_and_tag():
    import pipeline
    from config import SCORER_MODELS
    assert pipeline.tag_of(pipeline.MDP3) == "mdp3-e_32f"
    assert SCORER_MODELS["siglip"] == "google/siglip-so400m-patch14-384"
    from methods import REGISTRY
    assert "mdp3" in REGISTRY


def test_mdp3_select_algorithm():
    torch = pytest.importorskip("torch")  # torch chỉ có trên container Modal
    from methods.mdp3 import mdp3_select
    g = torch.Generator().manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(100, 16, generator=g, dtype=torch.float64), dim=-1)
    txt = torch.nn.functional.normalize(torch.randn(1, 16, generator=g, dtype=torch.float64), dim=-1)
    out = mdp3_select(img, txt, 32)
    assert out == sorted(set(out)) and len(out) == 32 and max(out) < 100
    assert mdp3_select(img[:20], txt, 32) == list(range(20))
