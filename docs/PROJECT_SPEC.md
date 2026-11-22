# PROJECT SPEC — Keyframe Selection cho Video-LLM trên UrbanVideo-Bench (Modal)

> Tài liệu chuẩn của dự án. Mọi thay đổi code phải giữ đúng các **hợp đồng dữ liệu** (mục 6) và **nguyên tắc so sánh công bằng** (mục 10). Nếu thay đổi kiến trúc, cập nhật file này trước.

---

## 1. Mục tiêu

Đánh giá xem các phương pháp **chọn keyframe** có giúp Video-LLM trả lời đúng hơn trên video drone/UAV hay không.

- **Benchmark:** UrbanVideo-Bench (ACL 2025): ~5.2k câu hỏi trắc nghiệm A–E trên ~1.5k video quay từ drone.
- **Phương pháp chính:** AKS, Adaptive Keyframe Sampling (CVPR 2025).
- **Baseline:** lấy đều (uniform) 32 frame, giống `run.py` gốc của UrbanVideo-Bench.
- **Mở rộng:** cắm thêm phương pháp khác (KTV, MoPrune, VETO, ...) vào cùng khung để so sánh.

Kết quả cuối là bảng accuracy (tổng và theo từng loại câu hỏi) cho mỗi tổ hợp *phương pháp × VLM*.

---

## 2. Nguồn gốc code

| Nguồn | Lấy gì | Bỏ gì |
|---|---|---|
| `github.com/ncTimTang/AKS` | Thuật toán ADA trong `frame_select.py` (port sang `methods/aks/ada.py`, giữ nguyên logic). Ý tưởng chấm điểm câu hỏi–frame trong `feature_extract.py`. | `outscores/`, `selected_frames/`, `datasets/` (dữ liệu LongVideoBench/VideoMME); `evaluation/` (các bản vá lmms-eval); `sevila.py`; phụ thuộc LAVIS/SeViLA/LLaVA-NeXT/lmms-eval |
| `github.com/EmbodiedCity/UrbanVideo-Bench.code` | Prompt trong `run.py` (copy nguyên văn vào `config.PROMPT`). Ý tưởng chấm đáp án bằng regex trong `eval.py`. | `question_generation/` (sinh câu hỏi bằng Gemini) |
| HuggingFace `EmbodiedCity/UrbanVideo-Bench` | `MCQ.parquet` + thư mục `videos/` | — |

**Viết mới:** toàn bộ hạ tầng Modal, chấm điểm bằng `transformers` (thay LAVIS), giao diện `FrameSelector` và registry, VLM qua vLLM, `evaluate.py`.

---

## 3. Kiến trúc pipeline

```
          ┌──────────────┐
BƯỚC 0    │ download.py  │  HF dataset + trọng số model ──► Volume
          └──────┬───────┘
                 ▼
          ┌──────────────────┐   methods/<tên>/   (uniform, aks, ...)
BƯỚC 1    │ select_frames.py │ ◄─────────────────
          └──────┬───────────┘
                 │  /vol/selections/<tag>.json   {Question_id: [frame idx]}
                 ▼
          ┌──────────────┐
BƯỚC 2    │  answer.py   │  VLM (vLLM) nhận PROMPT + câu hỏi + các frame đã chọn
          └──────┬───────┘
                 │  /vol/results/<tag>__<model>.csv
                 ▼
          ┌──────────────┐
BƯỚC 3    │ evaluate.py  │  (chạy local) accuracy tổng + theo question_category
          └──────────────┘
```

**Nguyên tắc tách bước:** việc chọn frame (bước 1) độc lập với VLM (bước 2).
- Một lựa chọn frame có thể dùng cho nhiều VLM.
- Mọi phương pháp được chấm bằng cùng một VLM và cùng một prompt.

---

## 4. Cấu trúc thư mục

```
urbanvideo-keyframe/
├── config.py          # hằng số: tên/đường dẫn Volume, model, PROMPT, selection_tag()
├── modal_infra.py     # Modal Volume + 3 Image (base / selector / vlm)
├── download.py        # BƯỚC 0
├── select_frames.py   # BƯỚC 1
├── answer.py          # BƯỚC 2
├── evaluate.py        # BƯỚC 3 (local, không cần Modal)
├── subset.py          # tạo tập câu cố định (outputs/subsets/), filter_mcq(), load_qids()
├── README.md          # hướng dẫn chạy ngắn
├── requirements.txt   # môi trường local (modal, pandas, opencv, pytest)
├── pytest.ini
├── tests/test_smoke.py  # test local với video giả lập: base, uniform, ADA, AKS(cache), extract_option
├── scripts/run_step.sh  # chạy 1 bước + bấm giờ -> outputs/logs/<bước>.log, outputs/logs/timing.tsv
├── docs/              # tài liệu: file này, RUN_LOG.md (nhật ký thời gian và kết quả từng lượt chạy)
└── methods/
    ├── __init__.py    # REGISTRY + get_method(name, **kwargs) — import lười
    ├── base.py        # FrameSelector (ABC), video_info(), read_frames()
    ├── uniform/__init__.py   # UniformSelector
    ├── aks/
    │   ├── __init__.py       # AKSSelector: ứng viên → chấm điểm (có cache) → ADA
    │   ├── scorer.py         # Scorer (clip|blip qua transformers), question_only()
    │   └── ada.py            # ada_select() — thuật toán judge-and-split
    └── _template/__init__.py # mẫu cho phương pháp mới
```

Thư mục `outputs/` được tạo trên máy local khi chạy và chứa bản sao kết quả: `outputs/selections/`, `outputs/results/`, `outputs/summary.csv`.

---

## 5. Lưu trữ trên Modal

| Volume | Mount | Nội dung |
|---|---|---|
| `urbanvideo-vol` | `/vol` | dataset, cache, selections, results |
| `hf-cache` | `/hf` (`HF_HOME`) | trọng số model HuggingFace |

```
/vol/data/UrbanVideo-Bench/MCQ.parquet
/vol/data/UrbanVideo-Bench/videos/<video_id>
/vol/cache/<method>/<scorer>/<Question_id>.json     # AKS: điểm câu hỏi–frame
/vol/selections/<tag>.json
/vol/results/<tag>__<model_short>.csv
```

**Image** (`modal_infra.py`):
- `base_image`: pandas, opencv, huggingface_hub. Dùng cho download, uniform, lưu file.
- `selector_image`: `base_image` cộng thêm torch, transformers. Dùng cho phương pháp cần GPU.
- `vlm_image`: vllm. Dùng cho bước trả lời.

Code local (`config`, `modal_infra`, `methods`) được upload qua `add_local_python_source(...)`. **Module mới ở cấp gốc phải được thêm vào `LOCAL_SRC`**, nếu không container sẽ không import được.

Sau khi ghi file lên Volume phải gọi `vol.commit()`.

---

## 6. Hợp đồng dữ liệu (KHÔNG được phá)

### 6.1 `MCQ.parquet` (đầu vào, từ dataset)
| Cột | Ý nghĩa |
|---|---|
| `Question_id` | id câu hỏi. Code luôn ép sang `str` |
| `video_id` | tên file video trong `videos/` (giả định đã gồm đuôi `.mp4`, giống `run.py` gốc) |
| `question_category` | loại câu hỏi (Duration, Causal, Landmark Position, ...) |
| `question` | thân câu hỏi **kèm các lựa chọn A–E** |
| `answer` | đáp án đúng |

### 6.2 Giao diện phương pháp — `FrameSelector.select()`
```python
def select(self, video_path: str, question: str, num_frames: int, qid=None) -> list[int]
```
- Trả về **chỉ số frame tuyệt đối trong video** (0-based, theo thứ tự đọc của OpenCV), **sắp xếp tăng dần**, không trùng, **tối đa `num_frames`** phần tử.
- `question` là chuỗi gốc có cả lựa chọn. Phương pháp tự quyết định có cắt bỏ phần lựa chọn hay không.
- `qid` dùng làm khóa cache.
- Constructor nhận `**kwargs`. `select_frames.py` luôn truyền `cache_dir=/vol/cache/<method>` cùng các tham số từ `--params`.

### 6.3 File chọn frame — `/vol/selections/<tag>.json`
```json
{ "<Question_id>": [12, 87, 140, ...], "<Question_id>": null }
```
Giá trị `null` nghĩa là câu đó bị lỗi khi chọn frame. Bước 2 sẽ bỏ qua câu đó, và nó bị tính là sai khi chấm.

Kết quả mỗi lượt được **gộp** vào file cũ cùng tag (cùng tag = cùng tham số, nên frame chọn ra như nhau). Nhờ vậy chạy tập con rồi chạy toàn bộ không mất kết quả.

Kèm theo là `/vol/selections/<tag>.timing.json`: `{"<Question_id>": {"s": 0.83, "cache_hit": false}}`. `s` là thời gian `select()` (không gồm nạp mô hình, xem `FrameSelector.warmup()`); `cache_hit = true` nghĩa là điểm lấy từ cache nên thời gian không có ý nghĩa; `null` nghĩa là phương pháp không có cache.

**Quy tắc đặt `tag`:** `selection_tag(method, num_frames, scorer)` cho ra `uniform_32f`, `aks-clip_32f`, `aks-blip_32f`. Có thể ghi đè bằng `--tag`. Nếu đổi siêu tham số (ví dụ `t1`), hãy đặt `--tag` riêng (như `aks-clip-t06_32f`) để không ghi đè kết quả cũ.

### 6.4 Cache điểm AKS — `/vol/cache/aks/<scorer>/<Question_id>.json`
```json
{ "scores": [0.21, 0.25, ...], "frames": [0, 30, 60, ...] }
```
Cache không phụ thuộc `t1`/`t2`/`all_depth`, nên đổi các tham số này không phải chấm lại. Nhưng cache **phụ thuộc** `cand_fps`: nếu đổi `cand_fps`, phải đổi `cache_dir` hoặc xóa cache.

### 6.5 Kết quả — `/vol/results/<tag>__<model_short>__<subset>.csv`
`<subset>` = tên file `--qids` không đuôi (ví dụ `subset_1000`), `head<N>` khi dùng `--limit N`, `full` khi chạy toàn bộ. File cũ chạy trước khi có `--qids` không có hậu tố; `evaluate.py` coi chúng là `head<n>`.

Các cột: `tag, model, subset, gpu` + mọi cột của MCQ + `Output` (văn bản VLM trả lời) + `n_frames` (số frame thực sự đưa vào) + `infer_s` (thời gian `LLM.chat` của batch chia đều cho số câu trong batch).

### 6.6 Tổng hợp — `outputs/summary.csv`
Mỗi dòng ứng với một file kết quả: `tag, model, subset, n, unparsed, frames, Overall, CI_lo, CI_hi, Macro, Recall, Perception, Reasoning, Navigation, src:<nguồn>, select_s, infer_s, <từng question_category>` (accuracy theo %).

`outputs/pairwise.csv`: mỗi cặp tag cùng `model` + `subset`: `n, acc_A, acc_B, diff, diff_lo, diff_hi` (CI hiệu số, paired bootstrap), `only_A, only_B, p` (McNemar exact), `p_holm`.

### 6.7 Tập câu hỏi — `outputs/subsets/<tên>.txt`
Mỗi dòng một `Question_id`. Tạo bằng `python subset.py` (phân tầng theo `question_category`, `seed = 0`): `subset_1000.txt` để báo cáo, `dev_300.txt` (không trùng) để dò siêu tham số. **Không sửa các file này sau khi đã có kết quả.**

---

## 7. Mô tả từng file

**`config.py`**: chỉ chứa hằng số, không import modal. `PROMPT` copy nguyên văn từ `run.py` gốc. Mặc định `DEFAULT_VLM = Qwen/Qwen2.5-VL-7B-Instruct`, `NUM_FRAMES = 32`.

**`download.py`**
- `download_dataset`: `snapshot_download` dataset về `/vol/data/...`, sau đó in số câu hỏi, số video và số video thiếu so với MCQ.
- `download_models(vlm)`: tải CLIP, BLIP và VLM vào `hf-cache`.
- Hai việc chạy song song (`spawn`).

**`select_frames.py`**
- Chia câu hỏi thành `--shards` phần (`df.iloc[shard::n_shards]`) và chạy song song bằng `starmap`.
- Phương pháp thuộc `CPU_METHODS` (`uniform`) chạy hàm `select_cpu`. Các phương pháp khác chạy `select_gpu` (mặc định T4, đổi bằng `--gpu`).
- Lỗi ở một câu không làm dừng cả lượt chạy: câu đó nhận giá trị `null`.
- Commit cache định kỳ mỗi 100 câu.

**`answer.py`**
- Class `VLM` (`@app.cls`) có các tham số `model`, `max_images`, `max_model_len`. Model được nạp một lần trong `@modal.enter`.
- Gom câu hỏi theo video để mỗi video chỉ đọc một lần. Mỗi frame được thu nhỏ cạnh dài về `--frame_size` (mặc định 448), mã hóa JPEG base64, rồi đưa vào vLLM qua `LLM.chat` theo batch.
- Tham số sinh: `temperature=0`, `max_tokens=256`.
- Nội dung message: text `PROMPT + "\n" + question` đứng trước, sau đó là các ảnh theo thứ tự thời gian (giống `run.py` gốc).

**`evaluate.py`**
- `extract_option()` lấy đáp án theo thứ tự ưu tiên: `Option: [X]`, rồi `X.`/`(X)` ở đầu dòng, rồi chữ in hoa đứng riêng. Nhận **A–H**: phần lớn câu có A–E, nhưng Action Generation có tới G (319 câu có đáp án F/G, 1 câu tới H). `PROMPT` gốc vẫn ghi "'A' to 'E'" và được giữ nguyên.
- Áp dụng cùng hàm này cho cột `answer` để chuẩn hóa đáp án đúng.
- Không đọc được đáp án thì tính là sai và đếm vào cột `unparsed`.

**`methods/base.py`**
- `video_info()`: trả về (tổng số frame, fps) qua OpenCV.
- `read_frames(path, indices)`: **đọc tuần tự** và giữ lại các frame cần. Cách này chậm hơn seek nhưng chính xác với mọi codec. Mọi phương pháp và `answer.py` đều phải dùng hàm này để chỉ số frame thống nhất.

---

## 8. Phương pháp AKS (chi tiết triển khai)

Quy trình cho mỗi câu hỏi (`methods/aks/__init__.py`):

1. **Frame ứng viên.** Lấy đều `n = duration × cand_fps` frame (mặc định 1 frame/giây). Nếu video ngắn thì lấy tối thiểu `2 × num_frames` frame, và không vượt quá tổng số frame.
2. **Chấm điểm** (`scorer.py`). Văn bản chấm là `question_only(question)`: bỏ phần lựa chọn bằng regex `(\n|\s)\(?A[\.\):]\s`, sau đó bỏ nhãn `Question:` ở đầu và `Choices:`/`Choice:`/`choose:` ở cuối. Câu hỏi trong dataset có dạng `Question: ...\nChoices:\nA. ...`. Có hai bộ chấm:
   - `clip` (`openai/clip-vit-base-patch32`): cosine similarity giữa vector câu hỏi và vector ảnh. Văn bản bị cắt ở 77 token.
   - `blip` (`Salesforce/blip-itm-large-coco`): xác suất "khớp" từ đầu ITM, tức `softmax(itm_score)[:, 1]`.
   - Kết quả được ghi vào cache (mục 6.4).
3. **ADA** (`ada.py`, port nguyên logic từ repo gốc):
   - Chuẩn hóa điểm về khoảng [0, 1].
   - Với mỗi đoạn: nếu `mean(top-n điểm) − mean(cả đoạn) > t1` và `std > t2`, giữ đoạn đó. Nếu không và `depth < all_depth`, chia đôi đoạn.
   - Một đoạn ở độ sâu `d` được lấy top `int(num_frames / 2^d)` frame theo điểm.
   - Gộp các frame, bỏ trùng, sắp xếp tăng dần.
   - Mặc định `t1 = 0.8`, `t2 = −100` (tức là bỏ qua điều kiện std), `all_depth = 5`.
   - Nếu số ứng viên ≤ `num_frames` thì trả về toàn bộ ứng viên.

Đổi tham số AKS qua `--params '{"scorer":"blip","t1":0.6,"all_depth":4,"cand_fps":2}'`.

**Khác biệt so với code gốc:**
- Chấm điểm bằng `transformers` thay vì LAVIS. Mô hình BLIP tương đương nhưng không đảm bảo trùng khớp từng con số.
- Có thêm `question_only()`.
- Có thêm số ứng viên tối thiểu cho video ngắn.
- Sửa lỗi kiểu dữ liệu của `t1`/`t2` (bản gốc khai báo `int`).

---

## 9. Thêm phương pháp mới

1. `cp -r methods/_template methods/<ten>`, đổi tên class, cài đặt `select()` đúng hợp đồng ở mục 6.2.
2. Thêm vào `REGISTRY` trong `methods/__init__.py`: `"<ten>": ("methods.<ten>", "<TenSelector>")`.
3. Nếu cần thư viện mới, thêm vào `selector_image` trong `modal_infra.py`. Nếu phương pháp không cần GPU, thêm tên vào `CPU_METHODS` trong `select_frames.py`.
4. Nếu phương pháp tốn chi phí tính toán, cache theo `qid` vào `self.cfg["cache_dir"]` rồi để `select_frames.py` commit.
5. Chạy thử: `modal run select_frames.py --method <ten> --limit 50`, sau đó `answer.py --tag <tag> --limit 50` và `python evaluate.py`.

**Không** sửa `answer.py` hay `evaluate.py` để phục vụ riêng một phương pháp. Mọi khác biệt phải nằm trong `methods/<ten>/`.

---

## 10. Nguyên tắc so sánh công bằng

Khi so sánh các phương pháp, các yếu tố sau phải **giống nhau**:
- Cùng VLM, cùng `PROMPT` (không chỉnh prompt), cùng `temperature=0`.
- Cùng `num_frames` (mặc định 32, bằng baseline gốc).
- Cùng `frame_size` và cùng tập câu hỏi (cùng `--limit`).

**Luôn chạy `uniform`** với cùng cấu hình làm mốc so sánh.

Ngoài accuracy, nên báo cáo thêm:
- Thời gian hoặc chi phí chọn frame (AKS tốn thêm bước chấm điểm).
- Accuracy theo từng `question_category`.

---

## 11. Lệnh chạy

```bash
pip install modal pandas pyarrow && modal setup
modal run download.py                                   # [--vlm <repo>] [--skip-models]
modal run select_frames.py --method uniform
modal run select_frames.py --method aks --params '{"scorer":"clip"}'
modal run answer.py --tag uniform_32f
modal run answer.py --tag aks-clip_32f
python evaluate.py
modal volume get urbanvideo-vol results outputs/        # (tuỳ chọn) lấy kết quả từ Volume
```

| Script | Tham số | Mặc định |
|---|---|---|
| `download.py` | `--vlm`, `--skip-models` | Qwen2.5-VL-7B |
| `select_frames.py` | `--method`, `--num-frames`, `--params` (JSON), `--shards`, `--limit`, `--qids`, `--tag`, `--gpu` | aks, 32, `{}`, 4, 0 (tất cả), —, tự sinh, T4 |
| `answer.py` | `--tag` (bắt buộc), `--model`, `--gpu`, `--shards`, `--limit`, `--qids`, `--frame-size`, `--batch`, `--max-images` | —, Qwen2.5-VL-7B, L40S, 1, 0, —, 448, 16, 32 |
| `subset.py` (local) | `--n`, `--dev`, `--seed` | 1000, 300, 0 |

Lưu ý: Modal chuyển tên tham số dấu `_` thành `-` trên CLI (ví dụ `--num-frames`).

---

## 12. Trạng thái và giả định chưa kiểm chứng

**Đã kiểm tra trên máy local** (với video giả lập):
- `UniformSelector`, `read_frames`, ADA, `AKSSelector` (đọc từ cache).
- `extract_option` và `evaluate.py`.
- Ba app Modal import được.

**Chưa kiểm tra** (cần xác nhận ở lần chạy đầu với `--limit 50`):
- [x] Chạy thật trên Modal (build image, Volume, GPU): chạy trọn pipeline `--limit 50` ngày 2026-10-26. Thời gian và kết quả ở `docs/RUN_LOG.md`.
- [x] Cấu trúc dataset trên HF: `MCQ.parquet` và `videos/` nằm ở gốc repo (2026-10-26): 5355 câu, 1394 video, **0 video thiếu**. Cột `Question_id` là int64, các cột còn lại là string.
- [x] `video_id` đã gồm đuôi file, ví dụ `EmbodiedCity_1.mp4`.
- [x] Regex của `question_only()` khớp 100% số câu. Đã bổ sung bước bỏ nhãn `Question:`/`Choices:`.
- [x] Cột `answer` là một chữ cái, **nhưng là A–G chứ không chỉ A–E** (Action Generation). Đã mở rộng `extract_option` sang A–H; sau khi sửa, 0/5355 đáp án không đọc được (trước khi sửa là 319).
- [x] Ghim phiên bản trong `modal_infra.py` (2026-10-26): `vllm==0.31.0`, `torch==2.13.0`, `transformers==5.17.0`, `numpy==2.3.5`, `opencv-python-headless==4.14.0.94`. Bộ này giải phụ thuộc OK bằng `uv pip compile` cho Linux/py3.11. Đã đối chiếu mã nguồn v0.31.0: vẫn có `LLM.chat`, `limit_mm_per_prompt`, `mm_processor_kwargs` và Qwen2.5-VL. **Chưa build thử trên Modal.**
- [x] Build 3 image trên Modal thành công với các phiên bản đã ghim. `vlm_image` cần thêm `VLLM_USE_FLASHINFER_SAMPLER=0`, vì `debian_slim` không có `nvcc` để FlashInfer JIT kernel sampler. Xem sự cố trong `docs/RUN_LOG.md`.
- [x] Ngân sách token: 32 ảnh × 448 px nằm trong `max_model_len = 32768`, mỗi ảnh 16:9 khoảng 144 token, mọi câu nhận đủ 32 frame. `max_pixels` chưa cần chỉnh.
- [x] GPU L40S đủ cho model 7B: trọng số chiếm 15,67 GiB, KV cache 19,23 GiB (360k token).

**Việc tiếp theo:**
- [x] Chạy baseline `uniform_32f` và `aks-clip_32f` trên toàn bộ dữ liệu, so sánh (Qwen2.5-VL-7B, 2026-10-26: 34,0 so với 34,1, p = 0,71).
- [ ] Thử `aks-blip_32f`, và AKS với các giá trị `t1` khác nhau.
- [ ] Đo thời gian chọn frame của từng phương pháp.
- [x] Cắm KTV (giai đoạn 1) — `methods/ktv`, 2026-11-18.
- [ ] InternVL2-8B × {Uniform, AKS+E, KTV+E} trên dev_300/subset_1000 (`pipeline.py`, `notebooks/experiment.ipynb`), xem giao thức mục 10.
- [ ] MoPrune (cần backend `transformers` riêng), VETO.

---

## 13. Quy tắc cho trợ lý AI làm việc trên repo này

1. Đọc file này trước khi sửa code. Giữ đúng các hợp đồng ở mục 6. Nếu buộc phải đổi, cập nhật mục 6 trong cùng lần thay đổi.
2. Không sửa logic `methods/aks/ada.py`. Nếu cần biến thể, tạo phương pháp mới (ví dụ `methods/aks_v2/`) và ghi rõ khác biệt.
3. Không chỉnh `config.PROMPT`, `temperature` hay cách chấm đáp án khi đang so sánh phương pháp (xem mục 10).
4. Mọi thử nghiệm mới chạy với `--limit 50` trước, rồi mới chạy toàn bộ.
5. Không ghi đè kết quả cũ: siêu tham số mới phải đi kèm `--tag` mới.
6. Module Python mới ở cấp gốc phải được thêm vào `LOCAL_SRC` trong `modal_infra.py`.
7. Khi hoàn thành một mục ở phần 12, đánh dấu `[x]` và ghi lại kết quả hoặc phát hiện.
