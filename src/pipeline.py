import re
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.ops as ops
from sentence_transformers import util
import clip

# ─── MATCH UTILS ────────────────────────────────────────────

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

def match_user_to_visible_objects(user_input, llava_response, object_groups, st_model):
    group_names = list(object_groups.keys())
    input_embedding = st_model.encode(user_input, convert_to_tensor=True)
    group_embeddings = st_model.encode(group_names, convert_to_tensor=True)

    cos_scores = util.cos_sim(input_embedding, group_embeddings)[0]
    best_index = torch.argmax(cos_scores).item()
    best_group = group_names[best_index]
    print(f"🔍 Matched user intent to group: {best_group}")

    object_list = object_groups[best_group]
    visible_objs = [obj for obj in object_list if is_object_mentioned(obj, llava_response)]

    return visible_objs, best_group


# ─── BOX UTILS ──────────────────────────────────────────────

def run_yolo(yolo_model, img_path, device):
    results = yolo_model.predict(img_path, verbose=False)
    boxes = results[0].boxes.xyxy
    return boxes.detach().to(device)

def nms_boxes(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    box_tensor = torch.stack(boxes)
    scores = torch.ones(len(boxes))
    keep_indices = ops.nms(box_tensor, scores, iou_threshold)
    return [boxes[i] for i in keep_indices]


# ─── CLIP REGION MATCHING ───────────────────────────────────

def detect_matching_boxes(model, processor, prompt, image_tensor, boxes, threshold=0.20, device="cuda"):
    height, width = image_tensor.shape[-2:]
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

            cropped = image_tensor[:, y_min:y_max+1, x_min:x_max+1]
            if cropped.numel() == 0:
                continue

            resized = processor(TF.to_pil_image(cropped)).unsqueeze(0).to(device)
            image_features = model.encode_image(resized)
            norm_image_features = F.normalize(image_features, p=2, dim=-1)
            similarity = torch.dot(norm_image_features.view(-1), norm_text_features.view(-1))

            if similarity.item() > threshold:
                matched_boxes.append(box.cpu())

    return nms_boxes(matched_boxes, iou_threshold=0.5)
