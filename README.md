# Keyframe selection trên UrbanVideo-Bench (chạy trên Modal)

So sánh các phương pháp chọn frame (uniform, AKS, ...) cho Video-LLM trên video UAV.
Mọi bước nặng chạy trên Modal. Chỉ bước chấm điểm chạy trên máy bạn.

## Cấu trúc

```
urbanvideo-keyframe/
├── config.py          # đường dẫn Volume, model mặc định, prompt gốc của UrbanVideo-Bench
├── modal_infra.py     # Volume + Image dùng chung trên Modal
├── download.py        # BƯỚC 0: tải dataset + trọng số model lên Volume (1 lần)
├── select_frames.py   # BƯỚC 1: chọn frame cho từng câu hỏi
├── answer.py          # BƯỚC 2: VLM (vLLM) trả lời bằng các frame đã chọn
├── evaluate.py        # BƯỚC 3: chấm accuracy, so sánh các phương pháp (chạy local)
└── methods/
    ├── base.py        # giao diện chung FrameSelector.select()
    ├── uniform/       # baseline lấy đều
    ├── aks/           # AKS: scorer.py (CLIP/BLIP) + ada.py (thuật toán ADA)
    └── _template/     # mẫu để thêm phương pháp mới
```

Trên Volume `urbanvideo-vol`:

```
/vol/data/UrbanVideo-Bench/   MCQ.parquet, videos/
/vol/cache/<method>/          điểm câu hỏi–frame (chạy lại không phải tính lại)
/vol/selections/<tag>.json    frame đã chọn
/vol/results/<tag>__<model>.csv
```

## Chạy

```bash
pip install modal pandas pyarrow
modal setup                                  # đăng nhập lần đầu

# 0. Tải dữ liệu và model (1 lần)
modal run download.py

# 1. Chọn frame (chạy thử với --limit 50 trước)
modal run select_frames.py --method uniform
modal run select_frames.py --method aks --params '{"scorer": "clip"}'

# 2. VLM trả lời
modal run answer.py --tag uniform_32f
modal run answer.py --tag aks-clip_32f

# 3. Chấm điểm và so sánh
python evaluate.py
```

### Chạy cả chuỗi, tắt máy vẫn chạy (InternVL2-8B: Uniform / AKS+E / KTV+E)

Điều khiển bằng `notebooks/experiment.ipynb`, hoặc gõ lệnh trực tiếp:

```bash
modal run pipeline.py --stage smoke                  # 20 câu, chờ xem kết quả
modal run --detach pipeline.py --stage night         # dò t1 trên dev_300 + 3 phương pháp trên subset_1000
modal volume get urbanvideo-vol status outputs/ --force    # xem tiến độ
modal volume get urbanvideo-vol results outputs/ --force   # rồi: python evaluate.py
```

Mọi bước gộp và lưu đều chạy trên Modal. Mỗi shard tự lưu kết quả lên Volume, nên chạy lại cùng lệnh sẽ bỏ qua phần đã xong.

Tùy chọn hay dùng:
- `--qids outputs/subsets/subset_1000.txt`: chạy trên tập con phân tầng cố định (tạo bằng `python subset.py`). Dùng tập này để báo cáo
- `--limit 50`: chạy thử trên 50 câu đầu (chỉ để thử, không dùng để báo cáo)
- `--shards 4`: chia việc cho 4 container song song
- `answer.py --model Qwen/Qwen2.5-VL-3B-Instruct --gpu A10G`: model nhỏ, GPU rẻ hơn. Nhớ tải model trước bằng `modal run download.py --vlm ...`
- `select_frames.py --params '{"scorer": "blip", "t1": 0.6, "all_depth": 4}'`: đổi tham số AKS

## Thêm phương pháp mới (KTV, MoPrune, ...)

1. Copy `methods/_template` thành `methods/ktv` và cài đặt hàm `select()`. Hàm nhận vào video và câu hỏi, trả về danh sách chỉ số frame.
2. Thêm `"ktv": ("methods.ktv", "KTVSelector")` vào `REGISTRY` trong `methods/__init__.py`.
3. Nếu cần thêm thư viện, thêm vào `selector_image` trong `modal_infra.py`.
4. Chạy `modal run select_frames.py --method ktv`, rồi `answer.py` và `evaluate.py` như trên.

Các bước 2 và 3 dùng chung cho mọi phương pháp, nên kết quả so sánh được trực tiếp.

## Nguồn

- AKS: Tang et al., CVPR 2025. Code gốc: github.com/ncTimTang/AKS. Thuật toán ADA được tách từ `frame_select.py`. Phần chấm điểm được viết lại bằng `transformers` thay cho LAVIS/SeViLA.
- UrbanVideo-Bench: Zhao et al., ACL 2025. Dataset: huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench. Prompt giữ nguyên theo `run.py` gốc.
