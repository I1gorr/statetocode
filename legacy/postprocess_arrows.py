import cv2
from ultralytics import YOLO
import math

# ========================================
# CONFIG
# ========================================

MODEL_PATH = "C:/statetocode/runs/detect/train/weights/best.pt"
IMAGE_PATH = "C:/statetocode/images/test1.png"
OUTPUT_PATH = "C:/statetocode/images/postprocessed.png"

CONF_THRESHOLD = 0.40

# How close arrow detections can be before
# we consider them parts of the same physical arrow
MAX_CENTER_DISTANCE = 120

# Minimum vertical overlap between boxes
MIN_Y_OVERLAP = 0.30


# ========================================
# HELPER FUNCTIONS
# ========================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_size(box):
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def vertical_overlap(a, b):
    """
    Returns vertical overlap ratio between two boxes.
    """

    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]

    overlap = max(0, min(ay2, by2) - max(ay1, by1))

    a_height = ay2 - ay1
    b_height = by2 - by1

    if a_height <= 0 or b_height <= 0:
        return 0

    return overlap / min(a_height, b_height)


def box_distance(a, b):
    """
    Distance between centers of two boxes.
    """

    ax, ay = box_center(a)
    bx, by = box_center(b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def boxes_should_merge(a, b):
    """
    Decide whether two arrow detections probably
    belong to the same physical arrow.
    """

    distance = box_distance(a, b)
    overlap = vertical_overlap(a, b)

    return (
        distance < MAX_CENTER_DISTANCE
        and overlap > MIN_Y_OVERLAP
    )


def merge_boxes(a, b):
    """
    Combine two bounding boxes.
    """

    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    ]


# ========================================
# LOAD MODEL
# ========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load image.")
    exit()

print("Image loaded.")


# ========================================
# RUN YOLO
# ========================================

results = model(
    IMAGE_PATH,
    conf=CONF_THRESHOLD,
    verbose=False
)

result = results[0]


# ========================================
# EXTRACT DETECTIONS
# ========================================

states = []
arrows = []

for box in result.boxes:

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    bbox = [
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    ]

    # Class 0 = state
    if cls == 0:
        states.append({
            "box": bbox,
            "confidence": conf
        })

    # Class 1 = arrow
    elif cls == 1:
        arrows.append({
            "box": bbox,
            "confidence": conf
        })


# ========================================
# PRINT RAW DETECTIONS
# ========================================

print()
print("RAW DETECTIONS")
print("================")

print(f"States: {len(states)}")
print(f"Arrows: {len(arrows)}")

for i, arrow in enumerate(arrows):
    print(
        f"Arrow {i}: "
        f"box={arrow['box']} "
        f"confidence={arrow['confidence']:.2f}"
    )


# ========================================
# MERGE ARROW DETECTIONS
# ========================================

merged = []

for arrow in arrows:

    current_box = arrow["box"]

    merged_into_existing = False

    for i in range(len(merged)):

        existing_box = merged[i]["box"]

        if boxes_should_merge(current_box, existing_box):

            merged[i]["box"] = merge_boxes(
                current_box,
                existing_box
            )

            merged[i]["confidence"] = max(
                merged[i]["confidence"],
                arrow["confidence"]
            )

            merged_into_existing = True
            break

    if not merged_into_existing:

        merged.append({
            "box": current_box,
            "confidence": arrow["confidence"]
        })


# ========================================
# PRINT MERGED RESULTS
# ========================================

print()
print("AFTER POST-PROCESSING")
print("======================")

print(f"States: {len(states)}")
print(f"Physical arrows: {len(merged)}")

for i, arrow in enumerate(merged):

    print(
        f"Arrow {i}: "
        f"box={arrow['box']} "
        f"confidence={arrow['confidence']:.2f}"
    )


# ========================================
# DRAW RESULTS
# ========================================

output = image.copy()

# Draw states
for i, state in enumerate(states):

    x1, y1, x2, y2 = state["box"]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    cv2.putText(
        output,
        f"STATE {i}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


# Draw merged arrows
for i, arrow in enumerate(merged):

    x1, y1, x2, y2 = arrow["box"]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        output,
        f"ARROW {i}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


# ========================================
# SAVE
# ========================================

cv2.imwrite(OUTPUT_PATH, output)

print()
print(f"Saved result to:")
print(OUTPUT_PATH)