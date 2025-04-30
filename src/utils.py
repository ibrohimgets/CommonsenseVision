import os
import json
import matplotlib.pyplot as plt

# ─── DRAW RESULTS ON IMAGE ────────────────────────────────

def save_boxes(image, labeled_boxes, save_path):
    """
    Draws bounding boxes with labels on the image and saves it.
    """
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


# ─── LOAD JSON CONFIG ────────────────────────────────────

def load_object_groups(json_path):
    """
    Loads the object trait groups from a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Trait group file not found: {json_path}")

    with open(json_path, "r") as f:
        object_groups = json.load(f)
    return object_groups
