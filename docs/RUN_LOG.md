# RUN LOG — nhật ký các lượt chạy

Mỗi bước chạy qua `bash scripts/run_step.sh <tên_bước> <lệnh...>`:
- Log đầy đủ: `outputs/logs/<tên_bước>.log`
- Thời gian máy đọc được: `outputs/logs/timing.tsv`
- Lượt lỗi: `outputs/logs/failed/`

Thời gian là **wall-clock tính từ máy local**, gồm cả thời gian build image, chờ container và nạp model.

---

## 2026-10-26 — Lượt thử đầu tiên `--limit 50`

**Cấu hình:** Qwen2.5-VL-7B-Instruct, 32 frame, `frame_size` 448, `temperature` 0, GPU chọn frame T4, GPU trả lời L40S, 4 shard chọn frame, 1 shard trả lời.
**Phiên bản:** vllm 0.31.0, torch 2.13.0, transformers 5.17.0 (xem `modal_infra.py`).

### Thời gian từng bước

| Bước | Lệnh | Thời gian | Ghi chú |
|---|---|---:|---|
| 0. Tải dữ liệu + model | `modal run download.py` | **446 s** (7,4 phút) | Build `base_image` 34 s. Dataset gồm 1398 file, tải mất 6 phút 19 giây; CLIP, BLIP và Qwen-7B tải song song |
| 1a. Chọn frame uniform | `select_frames.py --method uniform --limit 50` | **52 s** | CPU, 4 shard. 0 lỗi |
| 1b. Chọn frame AKS-clip | `select_frames.py --method aks --params '{"scorer":"clip"}' --limit 50` | **86 s** | Gồm 22,7 s build `selector_image`. T4, 4 shard. 0 lỗi |
| 2a. Trả lời uniform | `answer.py --tag uniform_32f --limit 50` | **227 s** | Nạp model 11,8 s, khởi tạo engine 49,1 s (biên dịch 16,5 s) |
| 2b. Trả lời AKS-clip | `answer.py --tag aks-clip_32f --limit 50` | **200 s** | Nạp model 10,0 s, khởi tạo engine 55,0 s (biên dịch 19,6 s) |
| 3. Chấm điểm | `python evaluate.py` | **1 s** | Local |

Bước 1a và 1b chạy song song; bước 2b bắt đầu khi 2a đang chạy.
Với 50 câu, phần lớn thời gian bước 2 là chi phí cố định: khởi động container, nạp model, khởi tạo engine (khoảng 60–70 s), không phải thời gian suy luận.

### Bộ nhớ GPU (L40S 48 GB)
- Trọng số model: 15,67 GiB
- KV cache: 19,23 GiB, tương đương 360.016 token, đủ cho khoảng 11 request dài 32k hoặc khoảng 70 câu dài khoảng 5k token chạy đồng thời
- Mọi câu nhận đủ 32 frame, không câu nào vượt `max_model_len`.

### Kết quả (n = 50; **chưa có ý nghĩa thống kê**)

| tag | n | unparsed | Overall | Causal | Counterfactual | Duration | Object Recall | Proximity | Scene Recall | Sequence Recall | Start/End Pos. | Traj. Captioning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aks-clip_32f | 50 | 0 | **50.0** | 75.0 | 66.7 | 63.6 | 60.0 | 50.0 | 50.0 | 100.0 | 28.6 | 33.3 |
| uniform_32f | 50 | 0 | **42.0** | 50.0 | 33.3 | 63.6 | 60.0 | 50.0 | 50.0 | 100.0 | 14.3 | 33.3 |

Lưu ý:
- Chênh lệch 8 điểm tương ứng **4 câu** trên 50, nên chưa đủ để kết luận.
- `--limit 50` lấy `df.head(50)`: 50 câu đầu chỉ gồm 29 video và **9/16 loại câu hỏi**. Không có Action Generation, Landmark Position hay Progress Evaluation, là các loại chiếm nhiều câu nhất.
- AKS và uniform chỉ chọn trùng trung bình **5%** số frame. AKS lấy frame theo cụm nên khoảng trống lớn nhất giữa 2 frame trung bình là 76 frame, so với 47 của uniform.

### Sự cố: lỗi `nvcc` ở bước 2 (đã sửa)
- **Triệu chứng:** `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`, xảy ra trong lúc vLLM khởi tạo, ở bước chạy thử sampler.
- **Nguyên nhân:** vLLM 0.31 mặc định dùng FlashInfer cho bước lấy mẫu top-k/top-p. FlashInfer cần biên dịch kernel lúc chạy (JIT) bằng `nvcc`, mà `debian_slim` không có.
- **Cách sửa:** thêm `VLLM_USE_FLASHINFER_SAMPLER=0` vào env của `vlm_image`. Vì `temperature=0` (greedy) nên kết quả không đổi.
- **Chi phí lãng phí:** 2 lượt lỗi chạy khoảng 37 và 38 phút (12:38 → 13:16), container L40S bị khởi động lại khoảng 28 lần mỗi lượt. Tổng cộng **khoảng 75 phút GPU L40S**; số tiền chính xác xem trên dashboard Modal. Log nằm trong `outputs/logs/failed/`.
- **Bài học:**
  1. Lỗi trong `@modal.enter` khiến Modal **khởi động lại container liên tục**, không tự dừng.
  2. Trên Windows, dừng tiến trình `bash` cha **không** dừng được `modal run` con. Phải dừng bằng `modal app stop -y <app_id>` và kiểm tra lại bằng `modal app list`.
  3. Chạy thử một lượt cho tới khi vượt qua bước khởi tạo engine, rồi mới chạy song song các lượt khác.

---

## 2026-10-26 — AKS-CLIP-32 trên `subset_1000`

**Cấu hình:** như lượt trước (Qwen2.5-VL-7B-Instruct, 32 frame, 448 px, `temperature` 0), tập câu `outputs/subsets/subset_1000.txt` (phân tầng 16 task, seed 0). Chọn frame: T4 × 8 shard. Trả lời: L40S × 4 shard.

| Bước | Lệnh | Thời gian | Ghi chú |
|---|---|---:|---|
| 1. Chọn frame AKS-CLIP | `select_frames.py --method aks --params '{"scorer":"clip"}' --qids outputs/subsets/subset_1000.txt --shards 8` | **425 s** | 0 lỗi. 990 câu chấm mới, 10 câu lấy cache. **1,98 s/câu** (đo trong code, không gồm nạp CLIP) |
| 2. Trả lời | `answer.py --tag aks-clip_32f --qids outputs/subsets/subset_1000.txt --shards 4` | **808 s** | 1000/1000 câu. **1,09 s/câu** (`infer_s`, thời gian `LLM.chat` chia đều theo batch) |

**Kết quả** (n = 1000, unparsed 0; 973 câu đủ 32 frame, 27 câu ít hơn do video ngắn / trùng frame; TB 31,7):

| Overall [CI 95%] | Macro | Recall | Perception | Reasoning | Navigation | AerialVLN | EmbodiedCity | RealWorld |
|---|---|---|---|---|---|---|---|---|
| **32.2** [29.4, 35.0] | 39.8 | 40.7 | 31.5 | 30.8 | 29.6 | 32.5 | 28.1 | 36.2 |

- Thấp hơn nhiều so với 50.0 ở lượt `--limit 50`: 50 câu đầu chỉ gồm các task dễ hơn và không có Action Generation / Landmark Position / Progress Evaluation (ở đây lần lượt 24.9 / 22.0 / 28.8).
- **Chưa so sánh được** vì chưa chạy Uniform-32 trên cùng `subset_1000`.

---

## 2026-10-26 — Uniform-32 và AKS-CLIP-32 trên toàn bộ 5.355 câu (kết quả chính)

**Cấu hình:** Qwen2.5-VL-7B-Instruct, 32 frame, 448 px, `temperature` 0, **H100** (`--gpu H100`, Modal cấp máy 140 GB), `--batch 32`, 8 shard, container 8 CPU, đọc trước 8 video song song với suy luận (`answer.py`, `PREFETCH`).

| Bước | Thời gian | Ghi chú |
|---|---:|---|
| Chọn frame uniform (CPU, 8 shard) | 261 s | 0 lỗi, 0,27 s/câu |
| Chọn frame AKS-CLIP (T4, 16 shard) | 941 s | 0 lỗi, 1,56 s/câu (4.315 câu chấm mới, 1.040 lấy cache) |
| Thử `--limit 50` trên H100 | 513 s | 48/50 câu cùng đáp án với L40S/batch 16, 2 câu khác (sai số số học khi đổi GPU/batch), nên mọi hàng so sánh phải cùng GPU + batch |
| Trả lời uniform | 1266 s | 5355/5355, `infer_s` 0,34 s/câu |
| Trả lời AKS | 1251 s | 5355/5355, `infer_s` 0,33 s/câu. Vài shard chờ GPU vì giới hạn số H100 chạy đồng thời |

**Kết quả** (unparsed 0; 158 câu < 32 frame ở cả hai hàng):

| | Overall [CI 95%] | Macro | Recall | Perception | Reasoning | Navigation |
|---|---|---|---|---|---|---|
| Uniform-32 | 34.0 [32.7, 35.2] | 42.4 | 40.7 | 35.1 | 34.3 | 30.3 |
| AKS-CLIP-32 | 34.1 [32.8, 35.3] | 42.8 | 40.6 | 35.5 | 35.1 | 30.1 |

AKS − Uniform = **+0.1 điểm** [−0.4, +0.7], chỉ AKS đúng 136 câu, chỉ Uniform đúng 129 câu, McNemar p = 0.71: **không khác biệt**.

**Vì sao gần như không khác:** frame AKS chọn gần như trùng với uniform.
- Mỗi frame AKS cách frame uniform gần nhất trung bình **0,23 bước** uniform. Khoảng trống lớn nhất giữa 2 frame là 1,62 bước, so với 1,09 bước của uniform.
- Chỉ 21,5% chỉ số frame trùng hẳn, nhưng phần còn lại chỉ lệch vài frame.
- Giải thích khả dĩ: với `t1 = 0.8` trên điểm đã chuẩn hóa [0, 1], điều kiện "top-K nổi bật hơn trung bình" hiếm khi thỏa, nên ADA chia đôi đến `all_depth = 5`, tức 2^5 = 32 đoạn, mỗi đoạn lấy 1 frame. Kết quả gần như lấy đều có xê dịch nhẹ.
- Cần kiểm chứng bằng cách đếm độ sâu dừng của ADA từ cache điểm.

Lượt `subset_1000` (L40S, batch 16) **không** dùng để so sánh với hai hàng trên.

**Sự cố:** phiên Claude Code bị khởi động lại giữa chừng nhưng tiến trình `modal run` trên máy vẫn chạy nên không mất gì. Lưu ý: tắt máy local khi `modal run` đang chạy sẽ mất toàn bộ kết quả, vì các shard chỉ trả kết quả về tiến trình local khi kết thúc.


---

## 2026-11-18 — Chuẩn bị InternVL2-8B trên tài khoản Modal mới (chưa chạy)

- Tài khoản/profile mới: `duonghoangminh15112005`. Volume `urbanvideo-vol` và `hf-cache` phải tạo và tải lại từ đầu (notebook, ô 2).
- Dùng lại từ máy local: cache điểm CLIP (`outputs/cache/clip`, 5.355 câu) và `uniform_32f.json`.
- Code mới: `methods/ktv`, biến thể +E, tên tag có t1/`-e`, chia shard theo video, mỗi shard tự lưu kết quả lên Volume (chạy lại sẽ bỏ qua shard đã xong), driver chạy trên Modal, `pipeline.py` chạy được `--detach`.
- Thứ tự chạy: `smoke` (20 câu, chờ xem) → `night` (detach: dò t1 trên dev_300, rồi 3 phương pháp trên subset_1000) → `full` (sau, nếu còn credit).
- Kết quả và thời gian ghi vào đây sau khi chạy.

### Sự cố 2026-11-19: vLLM treo khi khởi tạo InternVL2-8B (smoke, app `ap-2HjUtCHQuU4KWRe98pXNLI`)
- **Đã chạy được:** tải dataset + model (~7 phút); chọn frame `head20` cho uniform / AKS+E(t1 0.3, dùng cache CLIP) / KTV+E đều 0 lỗi, đủ 32 frame.
- **Triệu chứng:** 3 container H100 dừng sau dòng `Using V2 Model Runner` hơn 10 phút; GPU 0%, 595 MiB, load 0. `py-spy dump`: kẹt ở `UvaBuffer.__init__` → `torch.zeros(..., pin_memory=True)` (`vllm/v1/worker/gpu/buffer_utils.py:33`), trong `GPUModelRunner.__init__` của V2 Model Runner.
- **Thử 1 (không hiệu quả):** `VLLM_USE_V2_MODEL_RUNNER=0` → V1 runner vẫn treo, lần này ở `InputBatch.__init__` (`torch.zeros` CPU thường, ~64 MB).
- **Nguyên nhân thật** (`py-spy dump --native`): kẹt trong `libgomp` khi `fill_kernel` chạy song song → deadlock OpenMP sau fork. Tiến trình cha (bộ xử lý ảnh InternVL2, warmup) đã khởi tạo pool OpenMP, rồi vLLM fork EngineCore.
- **Cách sửa:** `VLLM_WORKER_MULTIPROC_METHOD=spawn` trong `vlm_image` (bỏ override V2 runner, giữ mặc định như lượt Qwen); `startup_timeout=15 phút` cho class `VLM`.
- Lỗi phụ cùng lượt: `select_driver` gọi `starmap([])` khi mọi shard đã xong → Modal chờ mãi; đã thêm `if todo:`.
- **Bài học:** luôn xem log/GPU sau ~5 phút khởi tạo engine; dừng app bằng `modal app stop -y <app_id>`.

### Sự cố 2026-11-19: lượt `night` bị hủy sau 5 phút (`modal run --detach`)
- Lượt `night` (app `ap-zyY9qYkJcKAAPTc0IlO67V`, 00:58) chỉ ghi xong bước chọn frame uniform rồi dừng. Log Modal: `Received a cancellation signal ... Successfully canceled input` lúc 18:03 UTC.
- **Nguyên nhân:** `modal run --detach` chạy nền trong phiên Claude Code; khi phiên kết thúc, tiến trình local bị tắt *bình thường* nên client Modal gửi tín hiệu HỦY cho `run_stage` đang chờ. `--detach` chỉ bảo vệ khi mất kết nối, không chặn lệnh hủy.
- **Không mất gì:** các `select_driver` đã spawn vẫn chạy xong — đủ phần chọn frame (uniform, KTV+E, AKS+E × 3 t1) trên dev_300 ∪ subset_1000. Bước trả lời chưa bắt đầu.
- **Cách sửa:** `launch.py` = `modal deploy pipeline.py` rồi `Function.from_name("uvb-pipeline", "run_stage").spawn(...)`, thoát ngay; app đã deploy chạy độc lập với máy local. Notebook ô 5/7 dùng `launch.py`.
- **Notebook:** mở lại máy thì kernel mới không còn biến → luôn chạy ô 0 trước ô 6 (lỗi `NameError: os`).

## 2026-11-19 — Lượt `night`: InternVL2-8B × {Uniform, AKS+E, KTV+E} trên subset_1000

**Cấu hình:** InternVL2-8B, 32 frame (1 tile 448 px/frame), H100, batch 32, vLLM 0.31 (`VLLM_WORKER_MULTIPROC_METHOD=spawn`). Phát lệnh bằng `python launch.py night` (deploy + spawn). Tổng thời gian 1.050 s (~17,5 phút) từ lúc spawn.

**Dò t1 trên dev_300 (AKS+E):** t1 = 0.8 → 33,00; 0.3 → 33,33; **0.2 → 34,33 (được chọn)**. Chênh tối đa 1,3 điểm (4/300 câu), nhỏ hơn nhiều so với sai số ±5 → t1 gần như không ảnh hưởng.

**Kết quả subset_1000** (unparsed 0; frame TB 31,68 ở cả 3 hàng):

| | Overall [CI 95%] | Macro | Recall | Perception | Reasoning | Navigation | AerialVLN | EmbodiedCity | RealWorld | Chọn frame (s/câu) |
|---|---|---|---|---|---|---|---|---|---|---|
| Uniform-32 | 34,3 [31,5; 37,2] | 42,9 | 41,3 | 39,9 | 31,9 | 27,6 | 34,8 | 31,2 | 36,7 | 0,05 |
| AKS+E-32 (t1 0.2) | 34,2 [31,3; 37,1] | 42,7 | 40,1 | 38,9 | 30,8 | 28,8 | 34,8 | 31,2 | 36,2 | ~0 (cache CLIP) |
| **KTV+E-32** | **35,3** [32,3; 38,1] | **43,8** | 41,9 | 40,8 | 31,9 | 29,1 | 35,2 | 32,3 | 38,9 | 7,6 |

So từng cặp (McNemar exact, Holm):
- KTV+E − Uniform = **+1,0** [0,0; +2,0], chỉ KTV đúng 18 câu / chỉ Uniform đúng 8 câu, p = 0,076 (Holm 0,23): **chưa có ý nghĩa thống kê**.
- AKS+E − Uniform = −0,1 [−1,5; +1,2], 23/24 câu, p = 1,0: **không khác**.
- AKS+E − KTV+E = −1,1, p = 0,15.

**Tái hiện baseline: KHÔNG ĐẠT.** Uniform-32 = 34,3 so với 28,1 của paper (lệch +6,2 > ngưỡng ±3).
- Không phải do trích đáp án: 100% output đúng định dạng `Option: [X]`; chấm kiểu chặt vẫn 34,3.
- Lệch lớn nhất ở nhóm Recall (Object Recall 58,3 so với 35,0; Sequence Recall 57,1 so với 32,2) và Causal (68,2 so với 33,6), các task này chỉ có 20–40 câu trong subset.
- Nguyên nhân có thể (chưa kiểm chứng): cách đưa frame vào InternVL2 (ảnh rời qua chat template vLLM, 1 tile) khác code đánh giá của paper; bản dataset HF (5.355 câu) khác bản paper (~5,2k); khác độ phân giải.
- InternVL2 thiên lệch chọn A (28% dự đoán so với 20% đáp án đúng).

## 2026-11-19 — Lượt `f8`: 8 frame, InternVL2-8B, subset_1000

**Cấu hình:** như `night` nhưng 8 frame. AKS: `all_depth = 3` (ADA lấy int(8/2^depth) frame/đoạn; all_depth = 5 cho 0 frame). KTV: `cand_k = 32` (cùng tập ứng viên và cache DINOv2 với lượt 32 frame). Phát lệnh `python launch.py f8`; tổng 2.391 s.

**Sự cố:** lần khởi tạo engine đầu tiên của 3 container trả lời dev_300 treo > 15 phút trong lúc Triton biên dịch kernel (`make_cubin` chờ tiến trình con không còn tồn tại); `startup_timeout` kết thúc và Modal khởi động lại, lần 2 chạy bình thường (~60 s). Tốn thêm ~15 phút × 3 H100 (~$3).

**Dò t1 (dev_300, 8 frame):** 0.8 → 34,00; **0.3 → 34,67 (chọn)**; 0.2 → 34,33.

**Kết quả subset_1000 (8 frame):** Uniform-8 34,1 [31,1; 37,0]; AKS+E-8 (t1 0.3) 34,0; KTV+E-8 34,2. Mọi cặp p = 1,0 (KTV − Uniform = +0,1; AKS − Uniform = −0,1).

**So với 32 frame:** Uniform-32 34,3 → Uniform-8 34,1 (gần như không giảm).

**Phát hiện chính — mô hình gần như không nhạy với frame:**
- Uniform-32 và Uniform-8 trả lời giống hệt nhau ở 89,8% số câu; cả 6 cấu hình (3 phương pháp × 2 ngân sách) giống nhau ở 77,2% số câu.
- Phân bố dự đoán lệch về A (29,5%) và C (23,4%).
- Giải thích khả dĩ: InternVL2-8B chủ yếu dựa vào câu hỏi/tiên nghiệm ngôn ngữ, ít dùng bằng chứng hình ảnh → thay đổi frame đầu vào khó làm thay đổi đáp án. Nên kiểm chứng bằng baseline không có video (0 frame) / 1 frame.

## 2026-11-19 — Baseline mù (0 frame, stage `blind`), InternVL2-8B, subset_1000

Cùng prompt + câu hỏi, không đưa frame nào (`textonly_0f`). 2 shard H100, engine khởi tạo bình thường (~60 s).

| | Overall [CI 95%] | Recall | Perception | Reasoning | Navigation | AerialVLN | EmbodiedCity | RealWorld |
|---|---|---|---|---|---|---|---|---|
| **0 frame** | **30,8** [27,9; 33,5] | 32,0 | 35,8 | 28,6 | 26,9 | 28,5 | 31,2 | 35,4 |
| Uniform-8 | 34,1 | 39,0 | 39,9 | 31,9 | 28,1 | 36,0 | 28,9 | 35,8 |
| Uniform-32 | 34,3 | 41,3 | 39,9 | 31,9 | 27,6 | 34,8 | 31,2 | 36,7 |
| KTV+E-32 | 35,3 | 41,9 | 40,8 | 31,9 | 29,1 | 35,2 | 32,3 | 38,9 |

- Không có video vẫn được 30,8 (ngẫu nhiên = 19,7) → phần lớn điểm đến từ câu hỏi/tiên nghiệm ngôn ngữ.
- Video chỉ thêm **+3,5 điểm** (Uniform-32 so với 0 frame: 102 so với 67 câu, McNemar p = 0,009); 67–69% câu trả lời giống hệt baseline mù.
- Video giúp chủ yếu ở Recall (Object +29, Sequence +19, Scene +19), Causal +14, Cognitive Map +10, Progress Evaluation +10; gần như không giúp Navigation (High-level Planning −4, Action Generation −4) và Proximity (−9).
- Theo nguồn: AerialVLN +6,3; EmbodiedCity 0; RealWorld +1,3.
- KTV+E-32 hơn baseline mù +4,5 (p = 0,0008), Uniform-32 +3,5 → KTV khai thác thêm ~1 điểm trong ~3,5 điểm mà video đóng góp (chênh KTV − Uniform vẫn chưa có ý nghĩa thống kê, p = 0,076).
- Số 28,1 của paper cho InternVL2-8B[32f] thấp hơn cả baseline mù của pipeline này (30,8) → khác biệt thiết lập (cách đưa frame, prompt, bản dataset) nhiều khả năng lớn hơn khác biệt do video.

## 2026-11-19 — Baseline mù cho Qwen2.5-VL-7B (subset_1000)

`python launch.py blind --model Qwen/Qwen2.5-VL-7B-Instruct` (H100, batch 32; tải Qwen lên hf-cache của tài khoản mới trước). So với kết quả có video của tháng 10 (Uniform-32, AKS-CLIP-32 trên full, cùng H100 batch 32), lọc đúng 1.000 câu của subset_1000.

| Mô hình | 0 frame | Uniform-32 | Video đóng góp | p (McNemar) | % câu giống 0 frame |
|---|---|---|---|---|---|
| InternVL2-8B | 30,8 | 34,3 | **+3,5** | 0,009 | 67,6 |
| Qwen2.5-VL-7B | 30,1 | 32,3 | **+2,2** | 0,088 | 72,7 |

- Qwen2.5-VL-7B dựa vào video **còn ít hơn** InternVL2-8B: video chỉ thêm +2,2 điểm (chưa có ý nghĩa thống kê), 72,7% câu trả lời giống hệt khi không có video. AKS-CLIP-32 của Qwen: +2,4 so với 0 frame.
- Đóng góp của video theo nhóm — InternVL2: Recall +9,3, Perception +4,0, Reasoning +3,3, Navigation +0,7; Qwen: đều ~+2 ở mọi nhóm.
- Kết luận: mức dựa vào video thấp là đặc điểm chung của các mô hình 7–8B trên UrbanVideo-Bench (với cách đưa frame của nhóm), không riêng InternVL2 → không nên chuyển sang Qwen để thử KTV.

## 2026-11-19 — LLaVA-NeXT-Video-7B @ 8 frame (stage `llava8`), subset_1000

**Cấu hình:** `llava-hf/LLaVA-NeXT-Video-7B-hf`, vLLM 0.31 (chỉ nhận 1 video/prompt): frame đã chọn → mảng (T, H, W, 3) RGB, cạnh dài 448; prompt `USER: <video>\n{PROMPT + câu hỏi} ASSISTANT:` (0 frame: `USER: {...} ASSISTANT:`); `max_model_len` 4096. Dùng lại đúng file chọn frame 8f của stage f8 (AKS t1 = 0.3 dò trên InternVL2). H100 batch 32; chạy thử 20 câu trước (8 frame/câu, 0 unparsed). Tổng 678 s.

| | Overall [CI 95%] | Hơn 0 frame (p) | Hơn Uniform-8 (p) | % giống 0 frame | Recall | Perception | Reasoning | Navigation |
|---|---|---|---|---|---|---|---|---|
| 0 frame | 25,0 [22,2; 27,5] | — | — | — | 29,1 | 21,8 | 28,6 | 25,0 |
| Uniform-8 | 26,8 [24,0; 29,5] | +1,8 (0,066) | — | 79,6 | 34,3 | 24,3 | 28,6 | 25,2 |
| AKS+E-8 | 26,4 | +1,4 (0,17) | −0,4 (0,60) | 79,3 | 32,6 | 24,6 | 27,5 | 25,0 |
| KTV+E-8 | 26,3 | +1,3 (0,19) | −0,5 (0,42) | 79,8 | 35,5 | 24,0 | 28,6 | 23,8 |

- Video đóng góp ít nhất trong 3 mô hình (+1,8, chưa có ý nghĩa thống kê); ~80% câu trả lời giống hệt 0 frame.
- **Thiên lệch vị trí rất mạnh:** chọn A ở 61% số câu (đáp án đúng là A chỉ 20%); 0 frame cũng 56,5% A. Độ chính xác khi chọn A 24,3% so với 30,8% khi chọn khác A.
- Thấp hơn nhiều so với 38,6 của paper (LLaVA-NeXT-Video-7B @ 32f): khác số frame (8 so với 32, ngữ cảnh 4096 không chứa đủ 32 × 144 token) và nhiều khả năng khác cách định dạng prompt → kết quả LLaVA nên coi là phụ, không dùng làm bằng chứng chính.
- Không phương pháp chọn frame nào hơn Uniform-8.

## 2026-11-19 — MDP3+E-32 (stage `mdp3`), InternVL2-8B, subset_1000

**Cấu hình:** `methods/mdp3` (port code chính thức sunh-23/MDP3; SigLIP so400m; λ = 0,2; đoạn 32; greedy vector hóa, trùng bản gốc 12/12 ca). Ứng viên chung với AKS/KTV, tối đa 128. +E. Chọn frame: T4 × 8 shard, 0 lỗi, 31,68 frame TB, **18,6 s/câu** khi tính mới (đọc video + SigLIP; 697 câu tính mới, còn lại dùng cache video). Trả lời: H100 × 4, batch 32. Tổng 3.421 s; lần khởi tạo engine đầu lại treo ~15 phút (startup_timeout, khởi động lại) → tốn thêm ~$3–4.

| | Overall [CI 95%] | Macro | Recall | Perception | Reasoning | Navigation | AerialVLN | EmbodiedCity | RealWorld |
|---|---|---|---|---|---|---|---|---|---|
| 0 frame | 30,8 | 37,1 | 32,0 | 35,8 | 28,6 | 26,9 | 28,5 | 31,2 | 35,4 |
| Uniform-32 | 34,3 | 42,9 | 41,3 | 39,9 | 31,9 | 27,6 | 34,8 | 31,2 | 36,7 |
| AKS+E-32 | 34,2 | 42,7 | 40,1 | 38,9 | 30,8 | 28,8 | 34,8 | 31,2 | 36,2 |
| KTV+E-32 | 35,3 | 43,8 | 41,9 | 40,8 | 31,9 | 29,1 | 35,2 | 32,3 | 38,9 |
| **MDP3+E-32** | **35,8** [33,1; 38,9] | **43,9** | **42,4** | **41,7** | 31,9 | **29,3** | **36,0** | **32,7** | 38,9 |

So từng cặp (McNemar exact; Holm trên 5 phép so sánh giữa các phương pháp):
- **MDP3+E − Uniform = +1,5** [+0,5; +2,6], 22 so với 7 câu, **p = 0,008, Holm 0,041 → có ý nghĩa thống kê** (lần đầu tiên trong dự án).
- MDP3+E − KTV+E = +0,5 [−0,5; +1,5], 15/10, p = 0,42 → không khác.
- MDP3+E − AKS+E = +1,6, 29/13, p = 0,020 (Holm 0,078).
- MDP3+E − 0 frame = +5,0 (p < 0,001); 68,1% câu giống 0 frame; chọn A 28,1%.
- Diễn giải: MDP3 (liên quan câu hỏi bằng SigLIP + đa dạng DPP + tuần tự) khai thác được ~5 trong phần video đóng góp, so với 3,5 của Uniform; hơn KTV chưa chứng minh được.

## 2026-11-21 — Qwen2.5-VL-7B: KTV+E-32 và MDP3+E-32 (subset_1000)

`modal run answer.py --tag {mdp3-e_32f|ktv-e_32f} --model Qwen/Qwen2.5-VL-7B-Instruct --gpu H100 --batch 32 --shards 2 --qids outputs/subsets/subset_1000.txt`. Dùng lại file chọn frame của lượt InternVL2 (không phụ thuộc mô hình). 1000/1000 câu, 0 unparsed, 99,9% đúng định dạng. MDP3 ~6 phút, KTV ~8 phút (1 container bị preempt, Modal tự chạy lại shard). Log: `outputs/logs/qwen_mdp3.log`, `qwen_ktv.log`. So với Uniform/AKS chạy 10/2026 (tài khoản cũ, cùng H100 batch 32, lọc đúng subset_1000).

| Qwen2.5-VL-7B, 32f | Overall [CI 95%] | Hơn Uniform [CI 95%] | thắng/thua | p | % giống Uniform |
|---|---|---|---|---|---|
| 0 frame | 30,1 [27,2; 32,9] | −2,2 | 65/87 | 0,088 | 72,7 |
| Uniform | 32,3 [29,4; 35,0] | — | — | — | — |
| AKS (t1 = 0,8, không +E) | 32,5 | +0,2 [−1,1; +1,6] | 26/24 | 0,89 | 91,0 |
| KTV+E | 31,6 | −0,7 [−2,1; +0,9] | 26/33 | 0,44 | 89,2 |
| MDP3+E | 31,9 | −0,4 [−1,7; +1,2] | 25/29 | 0,68 | 91,0 |

- Không phương pháp nào khác Uniform trên Qwen; phần "giảm" của KTV ở Reasoning dồn vào Causal (22 câu, 5 câu đổi). Phân bố đáp án A–E như nhau ở mọi phương pháp.
- Kết quả MDP3 > Uniform trên InternVL2 (+1,5, Holm 0,041) không lặp lại trên Qwen, nhất quán với việc Qwen ít dựa vào video hơn.
- Cập nhật `scripts/final_table.py` (xếp theo mô hình, thêm CI hiệu số, % đúng định dạng) và `scripts/make_summary_fig.py`.
