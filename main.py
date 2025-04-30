import os
import json
import re
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.ops as ops
import matplotlib.pyplot as plt
import clip
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bitsandbytes')

# ─── CONFIG ─────────────────────────────────────────
IMAGE_PATH = "data/classroom.png"
OBJECT_GROUPS_JSON = "configs/trait_group_with_pencil.json"
SAVE_IMAGE_PATH = "results/llava_gpt_result_yolo.jpg"
CLIP_BACKBONE = "ViT-B/16"
CLIP_THRESHOLD = 0.20
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs(os.path.dirname(SAVE_IMAGE_PATH), exist_ok=True)


# ─── LOAD IMAGE ─────────────────────────────────────
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = T.ToTensor()(image).to(DEVICE)

# ─── LOAD MODELS ────────────────────────────────────
clip_model, clip_preprocess = clip.load(CLIP_BACKBONE, device=DEVICE)
yolo_model = YOLO("yolov8x.pt")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=False)
print("✅ Models loaded")

# ─── LLaVA DESCRIPTION ──────────────────────────────
scene_prompt = "<image>\nUSER: Describe all visible visual elements in the image, including people, clothing, accessories, objects, and the setting. If there are people in the picture then please describe their outfit as well. Be short but thorough.\nASSISTANT:"
llava_inputs = llava_processor(text=scene_prompt, images=image, return_tensors="pt").to(DEVICE)
llava_output = llava_model.generate(**llava_inputs, max_new_tokens=256)
llava_response = llava_processor.tokenizer.decode(
    llava_output[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

print("\n" + "="*60)
print("🧠 LLaVA Description:\n", llava_response)
print("="*60 + "\n")

# ─── LOAD OBJECT GROUPS ─────────────────────────────
try:
    with open(OBJECT_GROUPS_JSON, "r") as f:
        object_groups = json.load(f)
except Exception as e:
    print(f"❌ Failed to load object groups JSON: {e}")
    exit()

group_names = list(object_groups.keys())

# ─── EMBEDDING SETUP ─────────────────────────────────
st_model = SentenceTransformer("all-MiniLM-L6-v2")
group_embeddings = st_model.encode(group_names, convert_to_tensor=True)

# ─── UTIL: STRICT OBJECT MENTION CHECK ──────────────
def is_object_mentioned(obj, llava_text):
    obj = obj.lower()
    llava_text = llava_text.lower()
    patterns = [
        fr"\b{re.escape(obj)}\b",
        fr"\b{re.escape(obj)}s\b",
        fr"\ba {re.escape(obj)}\b",
        fr"\ban {re.escape(obj)}\b",
    ]
    return any(re.search(pattern, llava_text) for pattern in patterns)

# ─── MATCH USER INTENT TO MULTIPLE OBJECTS ──────────
def get_detection_prompts():
    user_input = input("📝 What do you want to detect? ").strip().lower()
    input_embedding = st_model.encode(user_input, convert_to_tensor=True)

    # Print user input embedding

    cos_scores = util.cos_sim(input_embedding, group_embeddings)[0]
    best_index = torch.argmax(cos_scores).item()
    best_group = group_names[best_index]

    print(f"🔍 Matched user intent to group: {best_group}")

    object_list = object_groups[best_group]

    visible_objs = [obj for obj in object_list if is_object_mentioned(obj, llava_response)]

    if not visible_objs:
        print(" No related objects mentioned in the LLaVA description.")
        return [], ""

    print(f" Confirmed visible objects from LLaVA: {visible_objs}")
    return visible_objs, best_group

# ─── YOLO BOXES ─────────────────────────────────────
def extract_candidates_yolo(img_path):
    results = yolo_model.predict(img_path, verbose=False)
    boxes = results[0].boxes.xyxy
    return boxes.detach().to(DEVICE)

# ─── NMS FOR DUPLICATE CLEANUP ──────────────────────
def nms_boxes(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    box_tensor = torch.stack(boxes)
    scores = torch.ones(len(boxes))  # CLIP doesn't return scores, so use dummy
    keep_indices = ops.nms(box_tensor, scores, iou_threshold)
    return [boxes[i] for i in keep_indices]

# ─── CLIP MATCHING: MULTI-BOX ───────────────────────
def detect_all_matching_boxes(model, processor, prompt, image_pt, boxes, device=DEVICE):
    height, width = image_pt.shape[-2:]
    model.eval()
    matched_boxes = []

    with torch.no_grad():
        tokenized_query = clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(tokenized_query)
        norm_text_features = F.normalize(text_features, p=2, dim=-1)


        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            x_max, y_max = min(width, x_max), min(height, y_max)
            if x_min > x_max or y_min > y_max:
                continue

            cropped = image_pt[:, y_min:y_max+1, x_min:x_max+1]
            if cropped.numel() == 0:
                continue

            resized = processor(TF.to_pil_image(cropped)).unsqueeze(0).to(device)
            image_features = model.encode_image(resized)
            norm_image_features = F.normalize(image_features, p=2, dim=-1)
            similarity = torch.dot(norm_image_features.view(-1), norm_text_features.view(-1))

            if similarity.item() > CLIP_THRESHOLD:
                matched_boxes.append(box.cpu())

    return nms_boxes(matched_boxes, iou_threshold=0.5)

# ─── DRAW RESULT ────────────────────────────────────
def save_boxes(image, labeled_boxes, save_path):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for label, box in labeled_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, fontsize=10, color="white",
                bbox=dict(facecolor="red", alpha=0.5))
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# ─── MAIN ───────────────────────────────────────────
if __name__ == "__main__":
    visible_objects, group_label = get_detection_prompts()

    if not visible_objects:
        print(" No valid objects to detect. Exiting.")
        exit()

    candidate_boxes = extract_candidates_yolo(IMAGE_PATH)
    results = []

    for obj in visible_objects:
        matched_boxes = detect_all_matching_boxes(clip_model, clip_preprocess, obj, image_tensor, candidate_boxes)
        for box in matched_boxes:
            results.append((obj, box))
            print(f"Detected {obj} at box: {box.tolist()}")

    if results:
        save_boxes(image, results, SAVE_IMAGE_PATH)
        print(f"\n Detection complete!")
        print(f"Saved image: {SAVE_IMAGE_PATH}")
        print(f"Based on your intent, I looked for objects: {visible_objects}")
        print(f"Found and labeled {len(results)} object(s) in the image.")
    else:
        print(" No objects matched well enough")
