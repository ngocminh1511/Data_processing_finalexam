# Phương pháp đánh giá: Chọn frame cho Video-LLM trên UrbanVideo-Bench

> **Câu hỏi nghiên cứu:** Khi giữ nguyên mô hình và số frame, cách chọn frame nào giúp Video-LLM trả lời tốt nhất trên video UAV?
>
> **Nguyên tắc cốt lõi:** mỗi phép so sánh chỉ thay đổi **một yếu tố**, là phương pháp chọn frame (hoặc cắt token). Mô hình, số frame, prompt, độ phân giải, backend suy luận và bộ câu hỏi giữ nguyên.
>
> *Phiên bản 2 (2026-10-26): đã đối chiếu với code (`config.py`, `answer.py`, `evaluate.py`, `methods/`) và lượt chạy thử `--limit 50` trong [RUN_LOG.md](RUN_LOG.md). Thay đổi so với bản 1 ở mục 9.*

---

## 1. Tổng quan quy trình

```
Bước 0 – Kiểm tra pipeline (T1)
   Qwen2-VL-7B @0.25 fps  →  so với 36.2 trong bài UrbanVideo-Bench
        │  đạt (mục 3)
        ▼
Bước 1 – So sánh chính (T2): Qwen2.5-VL-7B, cùng 32 frame, backend vLLM
   Uniform-32 │ AKS-CLIP-32 │ KTV-32 (chỉ giai đoạn chọn frame)
        │
        ▼
Bước 2 – Mở rộng: phương pháp cắt token (cần backend riêng, mục 5)
   Uniform-32* │ MoPrune-32      Uniform-6* │ KTV-6
        │
        ▼
Bước 3 – Tổng hợp: bảng theo 16 task + bảng theo 4 nhóm + kiểm định thống kê
```

Bước 1 chạy được với pipeline hiện tại. Bước 2 cần sửa ở tầng mô hình (cắt token bên trong VLM), nên **không** làm được qua vLLM + giao diện `FrameSelector`; xem mục 5.

---

## 2. Thiết lập cố định (áp dụng cho mọi bước)

| Hạng mục | Giá trị | Nguồn trong code |
|---|---|---|
| Benchmark | UrbanVideo-Bench [1], bản trên HuggingFace `EmbodiedCity/UrbanVideo-Bench`: **5.355 câu, 1.394 video** (bài báo ghi 1.547 video, khoảng 5.2k câu). 16 task chia 4 nhóm (Recall, Perception, Reasoning, Navigation) | `download.py` |
| Mô hình chính | **Qwen2.5-VL-7B-Instruct**, zero-shot, không fine-tune | `config.DEFAULT_VLM` |
| Mô hình kiểm tra pipeline | Qwen2-VL-7B-Instruct, chỉ dùng ở Bước 0 vì bài [1] chỉ báo cáo số của dòng này | — |
| Backend | vLLM 0.31.0, `LLM.chat`, GPU L40S 48 GB | `answer.py`, `modal_infra.py` |
| Cách đưa frame | Mỗi frame là **một ảnh riêng** (`image_url`), sắp theo thứ tự thời gian, đặt sau phần text (giống `run.py` gốc). Mô hình **không** nhận timestamp | `answer.py` |
| Độ phân giải | Thu nhỏ cạnh dài về **448 px**, giữ tỉ lệ, mã hóa JPEG. Video 16:9 cho khoảng 144 token/frame, 32 frame khoảng 4.6k token | `--frame-size 448` |
| Prompt | Prompt trong `run.py` của repo code chính thức [1], chép nguyên văn vào `config.PROMPT`. Mỗi lần gọi chỉ hỏi 1 câu. Prompt ghi "'A' to 'E'" nhưng **giữ nguyên** dù Action Generation có tới G | `config.PROMPT` |
| Giải mã | Greedy (`temperature = 0`), tối đa 256 token output | `answer.py` |
| Trích đáp án | `extract_option()`, nhận A–H, theo thứ tự ưu tiên: `Option: X` → `X.`/`(X)` ở đầu dòng → chữ in hoa đứng riêng. Câu không parse được tính là **sai** và đếm vào cột `unparsed` | `evaluate.py` |
| Bộ câu hỏi | **Toàn bộ 5.355 câu**, dùng chung cho mọi hàng (mục 2.1) | — |
| Chỉ số | Accuracy theo từng task, Overall (micro) và Macro-Avg (mục 2.2) | `evaluate.py` |
| Phần cứng | Modal: chọn frame trên **T4** (CLIP nhỏ, GPU không phải nút thắt), trả lời trên **H100** (`--gpu H100`, container 8 CPU để đọc video song song). Cột `gpu` trong file kết quả ghi lại GPU đã dùng | `select_frames.py`, `answer.py` |

### 2.1. Bộ câu hỏi

- **Tập báo cáo: toàn bộ 5.355 câu** (chạy không có `--qids`, hậu tố file kết quả là `__full`). Mọi kết quả đưa vào báo cáo và slide đều lấy từ tập này, để so được với Table 3 của [1].
- **Tập thử nhanh: `outputs/subsets/subset_1000.txt`.** Đây là 1.000 câu phân tầng theo `question_category`, giữ tỉ lệ 16 task, `seed = 0`, tạo bằng `python subset.py`. Tập này chỉ dùng để kiểm tra một phương pháp mới chạy đúng trước khi chạy toàn bộ, **không** đưa vào báo cáo.
- **Tập dò siêu tham số: `outputs/subsets/dev_300.txt`** (300 câu, không trùng `subset_1000`). Chỉ cần khi thử tham số khác mặc định (mục 6.3).
- **Không dùng `--limit N` để báo cáo.** `--limit` lấy `df.head(N)`: 50 câu đầu chỉ có 29 video, đều là EmbodiedCity, và 9/16 task. `--limit` chỉ dùng để kiểm tra code chạy được.

### 2.2. Chỉ số

- **Overall (micro):** số câu đúng / tổng số câu. Đây là cột `Overall` mà `evaluate.py` đang tính.
- **Macro-Avg:** trung bình cộng accuracy của 16 task.
- Báo cáo **cả hai**, vì các task có số câu rất khác nhau và cần kiểm tra lại xem cột Avg trong Table 3 của [1] là micro hay macro trước khi so sánh với số trong bài.
- Theo nhóm (Recall, Perception, Reasoning, Navigation): accuracy trên toàn bộ câu thuộc nhóm (micro). *Việc cần làm:* thêm bảng ánh xạ task → nhóm vào `evaluate.py`.
- Luôn kèm cột `unparsed` và **số frame thực tế trung bình** (cột `n_frames` trong file kết quả) để kiểm tra mọi hàng nhận đúng số frame như nhau.

---

## 3. Bước 0 – Kiểm tra pipeline (T1)

**Mục đích:** bảo đảm phần đọc video, prompt và trích đáp án chạy đúng **trước khi** tốn GPU cho các phương pháp khác.

**Cấu hình:**
- Mô hình **Qwen2-VL-7B-Instruct** (không phải 2.5), vì [1] chỉ có số của dòng này.
- Lấy mẫu đều **0.25 fps**, nên số frame thay đổi theo độ dài video: video 10 giây có khoảng 3 frame, video 10 phút có 150 frame.
- *Việc cần làm trong code:*
  - Thêm chế độ lấy mẫu theo fps cho `uniform` (hiện chỉ có số frame cố định).
  - Chạy `answer.py` với `--max-images 160`.
  - Kiểm tra lại `max_model_len`: 150 frame × khoảng 144 token là khoảng 21.6k token, vẫn nằm trong 32.768.

**Tiêu chí đạt** (so với Avg = 36.2 của [1]):
- Chạy trên toàn bộ 5.355 câu: lệch **≤ 2 điểm**. Sai số chuẩn khoảng 0.7 điểm, nên lệch hơn 2 điểm khó do ngẫu nhiên.
- Chạy trên tập con 1.000 câu: lệch **≤ 3 điểm**. Khoảng tin cậy 95% khoảng ±3 điểm.
- `unparsed` < 1%.
- Không task nào lệch quá 15 điểm so với [1]. Một task lệch lớn thường là do lỗi parse hoặc lỗi đọc video riêng loại đó.

Nếu không đạt, kiểm tra theo thứ tự: regex trích đáp án → cách đọc video (`read_frames`) → độ phân giải → prompt.

**Lưu ý:** bản dataset trên HF (5.355 câu, 1.394 video) khác số trong bài (khoảng 5.2k câu, 1.547 video), nên một phần chênh lệch có thể đến từ khác biệt dữ liệu. Ghi rõ điều này nếu lệch nằm sát ngưỡng.

---

## 4. Bước 1 – So sánh chính (T2): cùng mô hình, cùng 32 frame

### 4.1. Các phương pháp

| Hàng | Phương pháp | Frame | Token | Cách chọn | Trạng thái code |
|---|---|---|---|---|---|
| A1 | **Uniform-32** | 32 | 100% | 32 frame cách đều trên toàn video (`np.linspace`) | Đã có (`uniform_32f`) |
| A2 | **AKS-CLIP-32** [2] | 32 | 100% | Chấm độ liên quan câu hỏi–frame bằng CLIP, chọn bằng thuật toán ADA (cân bằng liên quan và độ phủ) | Đã có (`aks-clip_32f`) |
| A3 | **KTV-32** [3] | 32 | 100% | K-means trên feature DINOv2 (m = 32), mỗi cụm lấy frame gần tâm nhất. Không dựa vào câu hỏi | Chưa có |
| A2b | AKS-BLIP-32 *(tuỳ chọn)* | 32 | 100% | Như A2 nhưng chấm bằng BLIP-ITM, gần với scorer của bài gốc hơn | Đã có (`aks-blip_32f`), chưa chạy |

### 4.2. Tập frame ứng viên dùng chung (cho AKS và KTV)

- Lấy đều **1 fps trên toàn video, tối thiểu 2 × K = 64 frame**, không vượt tổng số frame của video. Đây là cách `AKSSelector._candidates()` đang làm.
- KTV phải dùng **đúng hàm tạo ứng viên này**. Nên tách nó ra `methods/base.py` để hai phương pháp gọi chung.
- Lý do:
  1. AKS và KTV chọn từ cùng một tập đầu vào, nên so sánh công bằng.
  2. Giới hạn chi phí chạy DINOv2/CLIP với video dài.
  3. Video 10 giây ở 1 fps chỉ có 10 frame, không đủ để chọn 32.
- Không đổi `cand_fps` hay số ứng viên tối thiểu giữa chừng: cache điểm AKS phụ thuộc vào tập ứng viên.

### 4.3. Cấu hình chi tiết

**AKS-CLIP-32**
- Thuật toán ADA port nguyên logic từ code gốc, K = 32, `t1 = 0.8`, `t2 = −100`, `all_depth = 5`.
- Scorer: `openai/clip-vit-base-patch32` qua `transformers`, cosine similarity. Bài gốc dùng BLIP/SeViLA qua LAVIS, nên đây là **biến thể của nhóm**.
- Query đưa vào scorer chỉ là **phần thân câu hỏi** (`question_only()`: bỏ các lựa chọn và nhãn `Question:`/`Choices:`). Lý do: câu hỏi dài 50–250 từ, vượt giới hạn 77 token của CLIP. Phần dư vẫn bị cắt ở 77 token.

**KTV-32** *(biến thể do nhóm điều chỉnh)*
- Bài gốc dùng m = 6 để so sánh công bằng với IG-VLM và SF-LLaVA. Nhóm đặt m = 32 để cân bằng số frame, thuật toán giữ nguyên.
- Chỉ dùng **giai đoạn 1** (chọn frame), không cắt token, để chỉ khác Uniform và AKS ở cách chọn frame. Phần này cắm được qua `FrameSelector` như phương pháp bình thường.
- Không phụ thuộc câu hỏi nên cache **theo video**, không theo câu hỏi. Mọi câu cùng video nhận cùng tập frame.
- Các frame được chọn sắp xếp lại theo thứ tự thời gian.

### 4.4. Cách đọc kết quả Bước 1

| So sánh | Trả lời câu hỏi |
|---|---|
| A2 với A1 | Chọn frame **theo câu hỏi** có tốt hơn lấy đều không? |
| A3 với A1 | Chọn frame **đa dạng hình ảnh** (không theo câu hỏi) có tốt hơn lấy đều không? |
| A2 với A3 | Với video UAV, chọn theo câu hỏi hay theo độ đa dạng thì tốt hơn? |
| A2b với A2 | Đổi scorer CLIP → BLIP thay đổi bao nhiêu? *(chỉ khi chạy A2b)* |

Nên xem thêm kết quả theo task: AKS gom frame theo cụm (trong lượt thử, khoảng trống lớn nhất giữa 2 frame trung bình là 76 frame so với 47 của uniform), nên có thể **lợi** ở task hỏi về một vật thể/sự kiện cụ thể nhưng **thiệt** ở task cần toàn bộ quỹ đạo (Trajectory Captioning, Sequence Recall, Duration).

---

## 5. Bước 2 – Mở rộng: phương pháp cắt token

MoPrune và giai đoạn 2 của KTV cắt **token thị giác bên trong VLM**. Pipeline hiện tại không làm được việc này vì:
- Giao diện `FrameSelector` chỉ trả về chỉ số frame.
- `answer.py` chạy qua vLLM, không can thiệp được vào token sau vision encoder.
- PROJECT_SPEC mục 9 cấm sửa `answer.py` cho riêng một phương pháp.

**Yêu cầu nếu làm Bước 2:**
- Viết một backend trả lời riêng bằng `transformers` (ví dụ `answer_hf.py`), có hook cắt token sau vision encoder.
- **Chạy lại Uniform trên chính backend này** (hàng có dấu \*). Khác backend (vLLM ↔ transformers, kernel attention) có thể làm accuracy lệch, nên MoPrune và KTV-6 chỉ được so với hàng Uniform\* cùng backend.

| Hàng | Phương pháp | Frame | Token | Ghi chú |
|---|---|---|---|---|
| A1\* | Uniform-32 (backend HF) | 32 | 100% | Đối chứng cho A4 |
| A4 | **MoPrune-32** [4] | 32 (lấy đều) | 25% | r = 25%, τ = 0.8, η = 0.5 (mặc định bài gốc) |
| B0\* | Uniform-6 (backend HF) | 6 | 100% | Đối chứng cho B1 |
| B1 | **KTV-6** [3] | 6 | ≈ 27% | Tái hiện bài gốc: m = 6, α = 0.8, β kiểu *normal* |

**Điều chỉnh bắt buộc cho Qwen2.5-VL:**
- Encoder không có token [CLS]. Điểm quan trọng $S^{cls}$ của KTV được thay bằng cosine giữa từng token và token trung bình của frame. Ghi rõ trong báo cáo.
- Vì mỗi frame được đưa vào như **một ảnh riêng** (mục 2), mỗi frame có nhóm token riêng; encoder **không** gộp 2 frame liên tiếp như khi đưa vào dạng video. Motion score của MoPrune vì vậy tính **giữa các frame liền kề**, không phải giữa các nhóm 2 frame. Nếu chuyển sang đưa dạng video, phải ghi rõ và chạy lại cả hàng đối chứng.
- Khi cắt token, giữ nguyên vị trí (t, h, w) gốc của token còn lại cho M-RoPE.

**Cách đọc:**
- A4 với A1\*: cắt **75% token** thì độ chính xác mất bao nhiêu? *(Không dùng để kết luận MoPrune chọn frame tốt hơn, vì hai hàng dùng số token khác nhau.)*
- B1 chỉ so với B0\*, không so với các hàng 32 frame.

---

## 6. Bước 3 – Tổng hợp và kiểm định

### 6.1. Bảng đầy đủ (cột theo Table 3 của UrbanVideo-Bench)

*Recall:* TC = Trajectory Captioning · SR = Sequence Recall · OR = Object Recall · ScR = Scene Recall · S/E = Start/End Position
*Perception:* Prox = Proximity · Dur = Duration · LP = Landmark Position · GD = Goal Detection · CM = Cognitive Map
*Reasoning:* Cau = Causal · CF = Counterfactual · Asc = Association
*Navigation:* PE = Progress Evaluation · HLP = High-level Planning · AG = Action Generation

| # | Phương pháp | Mô hình | #Frame | #Token | Overall | Macro | TC | SR | OR | ScR | S/E | Prox | Dur | LP | GD | CM | Cau | CF | Asc | PE | HLP | AG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | Random [1] | — | — | — | 19.7 | | 18.5 | 17.0 | 20.8 | 13.5 | 21.8 | 37.8 | 35.6 | 19.7 | 18.0 | 21.9 | 18.2 | 25.0 | 18.3 | 21.8 | 15.9 | 16.4 |
| R1 | Uniform @0.25fps [1] | Qwen2-VL-7B | var | 100% | 36.2 | | 36.5 | 50.9 | 47.5 | 65.8 | 47.2 | 52.1 | 48.5 | 25.1 | 28.4 | 55.8 | 55.5 | 29.5 | 11.7 | 33.9 | 59.3 | 32.7 |
| R2 | Uniform @0.25fps (tái lập) | Qwen2-VL-7B | var | 100% | | | | | | | | | | | | | | | | | | |
| A1 | Uniform-32 | Qwen2.5-VL-7B | 32 | 100% | | | | | | | | | | | | | | | | | | |
| A2 | AKS-CLIP-32 | Qwen2.5-VL-7B | 32 | 100% | | | | | | | | | | | | | | | | | | |
| A3 | KTV-32 | Qwen2.5-VL-7B | 32 | 100% | | | | | | | | | | | | | | | | | | |
| A1\* | Uniform-32 (HF) | Qwen2.5-VL-7B | 32 | 100% | | | | | | | | | | | | | | | | | | |
| A4 | MoPrune-32 (HF) | Qwen2.5-VL-7B | 32 | 25% | | | | | | | | | | | | | | | | | | |
| B0\* | Uniform-6 (HF) | Qwen2.5-VL-7B | 6 | 100% | | | | | | | | | | | | | | | | | | |
| B1 | KTV-6 (HF) | Qwen2.5-VL-7B | 6 | ≈27% | | | | | | | | | | | | | | | | | | |

*R0 và R1 chép từ Table 3 của [1]; số Avg của bài ghi vào cột Overall, cần đối chiếu lại xem đó là micro hay macro (mục 2.2).*

### 6.2. Bảng tóm tắt (dùng cho slide)

| # | Phương pháp | #Frame | #Token | Recall | Perception | Reasoning | Navigation | Overall [CI 95%] | Chọn frame (s/câu) | Suy luận (s/câu) |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Uniform-32 | 32 | 100% | | | | | | | |
| A2 | AKS-CLIP-32 | 32 | 100% | | | | | | | |
| A3 | KTV-32 | 32 | 100% | | | | | | | |
| A4 | MoPrune-32 | 32 | 25% | | | | | | | |
| B1 | KTV-6 | 6 | ≈27% | | | | | | | |

### 6.3. Kiểm định và phân tích phụ

1. **Khoảng tin cậy:** Overall của mỗi hàng kèm khoảng tin cậy bootstrap 95% (1.000 lần lấy mẫu lại theo câu hỏi).
2. **So sánh từng cặp:**
   - Dùng McNemar test (exact) cho các cặp A1–A2, A1–A3, A2–A3, A1\*–A4 và B0\*–B1.
   - Có 5 phép so sánh nên hiệu chỉnh **Holm**; chênh lệch chỉ có ý nghĩa khi p đã hiệu chỉnh < 0.05.
   - Báo cáo thêm khoảng tin cậy của **hiệu số** bằng paired bootstrap (lấy mẫu lại cùng tập câu cho cả hai hàng).
3. **Mức chênh tối thiểu phát hiện được** (ước lượng, giả sử khoảng 30% số câu có kết quả khác nhau giữa hai phương pháp):

   | Số câu | Sai số chuẩn của hiệu số | Chênh tối thiểu để có ý nghĩa (≈ 2 SE) |
   |---|---|---|
   | 50 (lượt thử) | khoảng 7.7 điểm | khoảng 15 điểm |
   | 1.000 | khoảng 1.7 điểm | khoảng 3.5 điểm |
   | 5.355 | khoảng 0.75 điểm | khoảng 1.5 điểm |

   Vì vậy chênh 8 điểm ở lượt thử 50 câu (AKS 50.0 so với Uniform 42.0, tức 4 câu) **chưa nói lên điều gì**.
4. **Theo nguồn video:** tách kết quả theo nguồn video (EmbodiedCity, AerialVLN, video thật), giống mục 4.4 của [1]. Nguồn lấy từ tiền tố của `video_id` (ví dụ `EmbodiedCity_1.mp4`).
5. **Thời gian:**
   - Đo **trong code**, không dùng wall-clock của `timing.tsv`, vì số đó gồm cả build image, khởi động container và nạp model.
   - Chọn frame tính theo **s/câu** cho AKS (phụ thuộc câu hỏi) và **s/video** cho KTV (không phụ thuộc câu hỏi). Không tính lần nạp mô hình chấm điểm.
   - Suy luận tính theo s/câu, không tính khởi tạo engine.
   - Ghi kèm loại GPU: T4 cho chọn frame, L40S cho suy luận.
6. **Siêu tham số:**
   - Dùng giá trị mặc định của các bài gốc.
   - Nếu muốn thử `t1` hay scorer khác cho AKS, dò trên một **tập dev tách rời** (khoảng 300 câu, không trùng ID với tập báo cáo), rồi mới chạy tập báo cáo với giá trị đã chốt.
   - Mỗi cấu hình mới dùng `--tag` riêng.

---

## 7. Giới hạn cần nêu trong báo cáo

- Chỉ đánh giá trên **một mô hình** (Qwen2.5-VL-7B). Kết luận chưa chắc đúng với các họ mô hình khác. Bước 0 dùng Qwen2-VL-7B, nên chỉ chứng minh pipeline đúng, không phải số tham chiếu cho Qwen2.5-VL.
- Frame được đưa vào như danh sách ảnh **không có timestamp**. Với phương pháp chọn frame không đều (AKS, KTV), mô hình không biết khoảng cách thời gian giữa các frame. Điều này có thể làm các phương pháp này thiệt ở task về thời lượng và trình tự.
- Các điểm sau là **điều chỉnh của nhóm**, không phải cấu hình của bài gốc:
  - AKS dùng CLIP qua `transformers` thay vì BLIP/SeViLA qua LAVIS.
  - KTV-32 dùng m = 32.
  - Tập ứng viên dùng 1 fps, tối thiểu 64 frame.
  - KTV-6 thay [CLS] bằng token trung bình.
- MoPrune và KTV-6 chạy trên backend khác (transformers), dùng ít token hơn, nên không so sánh trực tiếp với các hàng của Bước 1.
- Dataset trên HF khác số liệu trong bài [1] (5.355 câu / 1.394 video so với khoảng 5.2k / 1.547).
- Trên toàn bộ 5.355 câu chỉ phát hiện được chênh lệch khoảng 1.5 điểm trở lên (mục 6.3). Accuracy theo từng task ít câu (khoảng 115–150 câu) vẫn dao động khoảng ±8 điểm.

---

## 8. Tài liệu tham khảo

[1] B. Zhao, J. Fang, Z. Dai, et al. **UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces.** *Proceedings of ACL 2025 (Long Papers)*. Anthology ID: 2025.acl-long.1558. arXiv: https://arxiv.org/abs/2503.06157. Code: https://github.com/EmbodiedCity/UrbanVideo-Bench.code

[2] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, Q. Ye. **Adaptive Keyframe Sampling for Long Video Understanding.** *CVPR 2025*, pp. 29118–29128. Code: https://github.com/ncTimTang/AKS

[3] B. Song, J. Peng, Y. Zhang, G. Chen, F. Yang, J. Guo. **KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs.** *AAAI 2026*, 40(11), pp. 9060–9068. DOI: 10.1609/aaai.v40i11.37862. Code: https://github.com/songbaiyang07-star/KTV (đã chuyển sang https://github.com/hktk07/KTV, bản clone ở `third_party/KTV`)

[4] W. Hong, Z. Wang, Y. Zhang, Z. Wang. **MoPrune: Scene-Guided Motion-Aware Token Pruning for Efficient Video Large Language Models.** *Findings of ACL 2026*, pp. 6929–6941. Anthology ID: 2026.findings-acl.344

---

## 9. Thay đổi so với bản 1

| Mục | Bản 1 | Bản 2 | Lý do |
|---|---|---|---|
| Mô hình chính | Qwen2-VL-7B | Qwen2.5-VL-7B; Qwen2-VL-7B chỉ dùng ở Bước 0 | Code và lượt chạy thử dùng Qwen2.5-VL (`config.DEFAULT_VLM`) |
| Độ phân giải | 360×420 | Cạnh dài 448 px, giữ tỉ lệ | Theo `answer.py --frame-size 448` |
| Prompt | "aerial agent", Appendix C.5 | Prompt `run.py` của repo chính thức | Theo `config.PROMPT` |
| Trích đáp án | Regex `Option: X` | Regex nhiều tầng, A–H | Action Generation có đáp án tới G/H |
| Số liệu dataset | 1.547 video, khoảng 5.2k câu | 5.355 câu, 1.394 video (bản HF) | Kiểm tra khi tải dữ liệu |
| Tập ứng viên | 1 fps, tối thiểu 128 | 1 fps, tối thiểu 2K = 64 | Theo `AKSSelector._candidates()`; cache đã tạo theo cách này |
| Tập con | Phân tầng khoảng 1.000 câu | Phân tầng 1.000 câu, seed cố định, lưu file ID; cấm dùng `--limit` để báo cáo | `--limit` lấy `head()`, chỉ có 9/16 task |
| Avg | Một cột Avg | Overall (micro) + Macro | Task có số câu rất khác nhau |
| Tiêu chí Bước 0 | Lệch ≤ 2 điểm | ≤ 2 (toàn bộ) / ≤ 3 (tập con), kèm kiểm tra theo task | Sai số chuẩn của tập con khoảng 1.5 điểm |
| MoPrune, KTV-6 | Nằm trong Bước 1 và 2 | Gộp vào Bước 2 "mở rộng", cần backend HF và hàng Uniform\* riêng | vLLM + `FrameSelector` không cắt token được |
| Ghi chú Qwen gộp 2 frame | Có | Bỏ; mỗi frame là một ảnh riêng | `answer.py` đưa frame dạng `image_url`, không phải video |
| Kiểm định | McNemar, p < 0.05 | McNemar + Holm, paired bootstrap, bảng mức chênh tối thiểu | 5 phép so sánh |
| Thời gian | s/video | Đo trong code; AKS theo s/câu, KTV theo s/video | `timing.tsv` gồm cả chi phí khởi động |

### Việc cần làm trong code để chạy đúng giao thức này

- [x] `--qids <file>` cho `select_frames.py` và `answer.py`, cùng script tạo tập con phân tầng có seed (`subset.py`).
- [ ] Chế độ `uniform` theo fps (Bước 0), và tải Qwen2-VL-7B-Instruct.
- [x] `evaluate.py`: thêm Macro-Avg, bảng theo 4 nhóm, theo nguồn video, `n_frames` trung bình, bootstrap CI, McNemar + Holm.
- [x] Đo thời gian chọn frame và suy luận trong code (`<tag>.timing.json`, cột `infer_s`).
- [x] Tách hàm tạo ứng viên ra `methods/base.py` (`candidate_frames`).
- [x] Cài `methods/ktv/` (giai đoạn 1, cache feature theo video). Xem mục 10.
- [ ] (Bước 2) Backend `answer_hf.py` có hook cắt token.


---

## 10. Phiên bản 3 (2026-11-18): InternVL2-8B, tái hiện baseline của paper

**Câu hỏi:** cùng mô hình và cùng 32 frame với một hàng baseline trong Table 3 của [1], AKS và KTV có hơn Uniform không?

| Hạng mục | Giá trị |
|---|---|
| Mô hình | **OpenGVLab/InternVL2-8B**. Paper có đúng hàng InternVL2-8B[32f] = **28,1** (Avg là micro: hàng Random có trung bình 16 task là 21,3 nhưng bảng ghi 19,7) |
| Đưa frame | 32 ảnh, mỗi ảnh 1 tile 448 px (`max_dynamic_patch = 1`) = 256 token. `max_model_len = 16384`. Prompt, cách trích đáp án và greedy giữ nguyên mục 2 |
| Backend | vLLM 0.31.0, H100, batch 32 |
| Hàng so sánh | **Uniform-32** (`uniform_32f`), **AKS+E-32** (`aks-clip-t<t1>-e_32f`), **KTV+E-32** (`ktv-e_32f`) |
| Tập câu | Dò t1 trên `dev_300`, báo cáo trên `subset_1000` trước. Toàn bộ 5.355 câu chạy sau nếu còn ngân sách |
| Điều khiển | `notebooks/experiment.ipynb` → `pipeline.py` (`--stage smoke/night/full`), chạy `--detach` |

**Biến thể +E (frame đầu và cuối):**
- Uniform (`linspace(0, total-1)`) luôn có frame đầu và cuối, còn AKS/KTV thường không có. +E gỡ bất lợi này cho các câu về điểm đầu/cuối, tiến độ và quỹ đạo.
- Cách làm (`methods/base.py: add_endpoints`): phương pháp chọn đủ 32 frame → thêm frame 0 và `total-1` → bỏ dần frame ở giữa có khoảng cách tới frame kề nhỏ nhất (thừa nhất) cho tới khi còn 32.
- Áp dụng cho **mọi câu**, không dựa vào nhãn `question_category`.
- Không chạy ADA với K = 30: ở độ sâu 5, `k = int(30/32) = 0` nên ADA không chọn được frame nào.

**Chọn t1 của AKS:** dò t1 ∈ {0.8, 0.3, 0.2}. Phân tích cache cho thấy với t1 ≥ 0.5, 100% đoạn bị chia tới độ sâu 5 (≈ lấy đều); t1 = 0.3 thì 15% câu dừng ở độ sâu 0; t1 = 0.2 thì 55%. **Quy tắc ghi trước khi chạy:** chọn t1 có Overall cao nhất trên `dev_300`; nếu bằng nhau thì lấy t1 lớn hơn (gần mặc định bài gốc hơn). Không chỉnh t1 trên `subset_1000`.

**KTV giai đoạn 1** (`methods/ktv`, port từ `hktk07/KTV`):

| | Bài gốc | Bản của nhóm |
|---|---|---|
| Feature | DINOv2-large, `pooler_output` (CLS), Resize 256 bicubic → CenterCrop 224 → chuẩn hóa ImageNet | Giữ nguyên |
| Ứng viên | Tối đa 5.400 frame lấy đều | `candidate_frames()` (1 fps, tối thiểu 64), giống AKS |
| Phân cụm | KMeans(k, k-means++, n_init = 10, random_state = 0) | Giữ nguyên, k = 32 |
| Chọn frame | Frame gần tâm cụm nhất; nếu trùng thì bỏ hết, quay về lấy đều | Nếu trùng thì lấy frame gần kế tiếp chưa dùng |
| Thứ tự | Xếp lại theo CLIP | Theo thời gian (bỏ bước xếp lại) |
| Cắt token | Có (giai đoạn 2) | Không, chỉ giai đoạn 1 |

**Tiêu chí tái hiện:** Uniform-32 trên `subset_1000` lệch ≤ 3 điểm so với 28,1 (mục 3). Nếu không đạt thì dừng lại sửa pipeline, chưa so sánh phương pháp.

**Giới hạn thêm:** cách đưa frame vào InternVL2 (ảnh rời qua chat template của vLLM) có thể khác code đánh giá của [1]. Nếu lệch, đó là một nguyên nhân cần kiểm tra trước tiên.

---

## 11. Bổ sung (2026-11-19): MDP3 [5]

**MDP3+E-32** (`methods/mdp3`, port từ code chính thức `github.com/sunh-23/MDP3`, Apache-2.0, clone ở `third_party/MDP3`):

| | Code gốc | Bản của nhóm |
|---|---|---|
| Đặc trưng | SigLIP so400m-patch14-384 (frame + câu hỏi; câu hỏi > 64 token cắt đoạn rồi lấy trung bình) | Giữ nguyên |
| Kernel / mục tiêu | Tổng 5 kernel Gaussian (α = 2⁻³…2¹); (1/λ)·2·Σ log r + log det(L_S), λ = 0,2 | Giữ nguyên |
| Tuần tự | Đoạn 32 frame, quy hoạch động qua các đoạn, điều kiện trên 1 frame cuối của đoạn trước | Giữ nguyên; đoạn cuối được phép ngắn hơn 32 |
| Chọn tham lam | Lặp từng ứng viên, cập nhật Cholesky | **Vector hóa** (một lần giải tam giác cho mọi ứng viên); kiểm tra trùng kết quả bản gốc 12/12 ca, nhanh ~15 lần |
| Ứng viên | 128 frame lấy đều | Tập chung với AKS/KTV (1 fps, tối thiểu 64), lấy đều xuống tối đa 128 |
| Số frame | 8 | 32, kèm +E |
| Prompt cho SigLIP | `clear_prompt()` theo benchmark của paper | Câu hỏi kèm lựa chọn (không có phần hướng dẫn) |

Chạy trên InternVL2-8B, subset_1000, không dò tham số; so với Uniform-32 / AKS+E-32 / KTV+E-32 (McNemar).

[5] H. Sun et al. **MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs.** *ICCV 2025*. arXiv: 2501.02885. Code: https://github.com/sunh-23/MDP3
