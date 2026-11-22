"""Cấu hình chung: đường dẫn trên Volume, model mặc định, prompt."""

# ---- Volume trên Modal ----
VOL_NAME = "urbanvideo-vol"     # dataset + cache điểm + frame đã chọn + kết quả
HF_VOL_NAME = "hf-cache"        # trọng số model tải từ HuggingFace
VOL_PATH = "/vol"
HF_PATH = "/hf"

DATA_DIR = f"{VOL_PATH}/data/UrbanVideo-Bench"
VIDEO_DIR = f"{DATA_DIR}/videos"
MCQ_PATH = f"{DATA_DIR}/MCQ.parquet"
CACHE_DIR = f"{VOL_PATH}/cache"            # điểm câu hỏi–frame (AKS...)
SELECTION_DIR = f"{VOL_PATH}/selections"   # frame đã chọn, mỗi phương pháp 1 file
RESULT_DIR = f"{VOL_PATH}/results"         # câu trả lời của VLM

# ---- Dataset / model ----
HF_DATASET = "EmbodiedCity/UrbanVideo-Bench"
SCORER_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "blip": "Salesforce/blip-itm-large-coco",
    "dinov2": "facebook/dinov2-large",   # feature cho KTV (giống repo gốc hktk07/KTV)
    "siglip": "google/siglip-so400m-patch14-384",  # MDP3 (giống code gốc sunh-23/MDP3)
}
DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_FRAMES = 32
# Baseline "mù": cùng prompt + câu hỏi nhưng KHÔNG đưa frame nào (đo mức mô hình dựa vào video).
TEXT_ONLY_TAG = "textonly_0f"

# ---- Prompt: giữ nguyên run.py gốc của UrbanVideo-Bench để so sánh công bằng ----
PROMPT = (
    "This video (captured into multiple frames of images as follows) presents the perception "
    "data of an agent moving in the environment from a first person perspective. Please answer "
    "the following questions: \n"
    "The template for the answer is: \n"
    "Option: []; Reason: []\n"
    "where the Option only outputs one option from 'A' to 'E' here, do not output redundant "
    "content. Reason explains why you choose this option."
)


# ---- 16 task -> 4 nhóm, theo thứ tự cột Table 3 của UrbanVideo-Bench ----
TASK_GROUPS = {
    "Recall": ["Trajectory Captioning", "Sequence Recall", "Object Recall", "Scene Recall",
               "Start/End Position"],
    "Perception": ["Proximity", "Duration", "Landmark Position", "Goal Detection", "Cognitive Map"],
    "Reasoning": ["Causal", "Counterfactual", "Association Reasoning"],
    "Navigation": ["Progress Evaluation", "High-level Planning", "Action Generation"],
}


def selection_tag(method: str, num_frames: int, params: dict | None = None) -> str:
    """Tên file kết quả chọn frame, ghi đủ tham số khác mặc định để không ghi đè nhau.

    uniform_32f, aks-clip_32f (t1 mặc định), aks-clip-t0.3-e_32f, ktv-e_32f
    (-e = biến thể +E: luôn có frame đầu và cuối)."""
    p = params or {}
    name = method
    if p.get("scorer"):
        name += f"-{p['scorer']}"
    if "t1" in p:
        name += f"-t{p['t1']:g}"
    if p.get("endpoints"):
        name += "-e"
    return f"{name}_{num_frames}f"
