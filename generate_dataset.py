import cv2
import numpy as np
import os
import random
import math
import shutil


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "C:/statetocode/dataset"

NUM_IMAGES = 300

IMAGE_SIZE = 640

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# State appearance
MIN_RADIUS = 35
MAX_RADIUS = 55

# Arrow appearance
ARROW_THICKNESS = 5
ARROW_HEAD_LENGTH = 22
ARROW_HEAD_ANGLE = 30  # degrees

# Minimum distance between state centers
MIN_STATE_DISTANCE = 220


# ============================================================
# CREATE DATASET DIRECTORIES
# ============================================================

directories = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test"
]

for directory in directories:

    path = os.path.join(
        OUTPUT_DIR,
        directory
    )

    os.makedirs(path, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def generate_state_positions(num_states):

    positions = []

    attempts = 0

    while len(positions) < num_states and attempts < 1000:

        attempts += 1

        x = random.randint(100, IMAGE_SIZE - 100)
        y = random.randint(100, IMAGE_SIZE - 100)

        candidate = (x, y)

        valid = True

        for existing in positions:

            if distance(candidate, existing) < MIN_STATE_DISTANCE:
                valid = False
                break

        if valid:
            positions.append(candidate)

    return positions


def point_on_circle(center, radius, angle):

    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)

    return int(x), int(y)


# ============================================================
# DRAW ARROW
# ============================================================

def draw_arrow(image, start, end):

    """
    Draw a straight arrow from start to end.

    Returns the bounding box of the arrow:

        x_min, y_min, x_max, y_max
    """

    # Draw shaft
    cv2.line(
        image,
        start,
        end,
        (0, 0, 0),
        ARROW_THICKNESS,
        cv2.LINE_AA
    )

    # Direction of arrow
    dx = start[0] - end[0]
    dy = start[1] - end[1]

    angle = math.atan2(dy, dx)

    # Arrowhead directions
    angle1 = angle + math.radians(ARROW_HEAD_ANGLE)
    angle2 = angle - math.radians(ARROW_HEAD_ANGLE)

    p1 = (
        int(end[0] + ARROW_HEAD_LENGTH * math.cos(angle1)),
        int(end[1] + ARROW_HEAD_LENGTH * math.sin(angle1))
    )

    p2 = (
        int(end[0] + ARROW_HEAD_LENGTH * math.cos(angle2)),
        int(end[1] + ARROW_HEAD_LENGTH * math.sin(angle2))
    )

    # Draw arrowhead
    cv2.line(
        image,
        end,
        p1,
        (0, 0, 0),
        ARROW_THICKNESS,
        cv2.LINE_AA
    )

    cv2.line(
        image,
        end,
        p2,
        (0, 0, 0),
        ARROW_THICKNESS,
        cv2.LINE_AA
    )

    # Calculate bounding box
    points = np.array([
        start,
        end,
        p1,
        p2
    ])

    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])

    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # Padding for line thickness
    padding = ARROW_THICKNESS + 3

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)

    x_max = min(IMAGE_SIZE - 1, x_max + padding)
    y_max = min(IMAGE_SIZE - 1, y_max + padding)

    return x_min, y_min, x_max, y_max


# ============================================================
# YOLO BOUNDING BOX CONVERSION
# ============================================================

def bbox_to_yolo(x_min, y_min, x_max, y_max):

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    width = x_max - x_min
    height = y_max - y_min

    # Normalize
    x_center /= IMAGE_SIZE
    y_center /= IMAGE_SIZE

    width /= IMAGE_SIZE
    height /= IMAGE_SIZE

    return x_center, y_center, width, height


# ============================================================
# GENERATE ONE IMAGE
# ============================================================

def generate_image():

    image = np.ones(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    ) * 255

    # Random number of states
    num_states = random.randint(2, 4)

    positions = generate_state_positions(num_states)

    # If we couldn't place enough states, retry
    if len(positions) != num_states:
        return None, None

    # Random radius for each state
    radii = []

    for _ in range(num_states):

        radii.append(
            random.randint(
                MIN_RADIUS,
                MAX_RADIUS
            )
        )

    # --------------------------------------------------------
    # Draw states
    # --------------------------------------------------------

    labels = []

    for (x, y), radius in zip(positions, radii):

        cv2.circle(
            image,
            (x, y),
            radius,
            (0, 0, 0),
            5,
            cv2.LINE_AA
        )

        # State bounding box
        x_min = x - radius
        y_min = y - radius
        x_max = x + radius
        y_max = y + radius

        labels.append(
            (
                0,
                x_min,
                y_min,
                x_max,
                y_max
            )
        )

    # --------------------------------------------------------
    # Initial arrow
    # --------------------------------------------------------

    initial_state = random.randint(
        0,
        num_states - 1
    )

    state_center = positions[initial_state]
    state_radius = radii[initial_state]

    # Random direction pointing INTO the state
    angle = random.uniform(
        0,
        2 * math.pi
    )

    # Arrow ends at circle boundary
    end = point_on_circle(
        state_center,
        state_radius,
        angle
    )

    # Start farther away
    start_distance = state_radius + random.randint(
        70,
        120
    )

    start = point_on_circle(
        state_center,
        start_distance,
        angle
    )

    arrow_bbox = draw_arrow(
        image,
        start,
        end
    )

    labels.append(
        (
            1,
            *arrow_bbox
        )
    )

    # --------------------------------------------------------
    # Transitions
    # --------------------------------------------------------

    # Create a random chain so every diagram has
    # at least some meaningful connectivity.

    for i in range(num_states - 1):

        source = i
        target = i + 1

        source_center = positions[source]
        target_center = positions[target]

        source_radius = radii[source]
        target_radius = radii[target]

        dx = target_center[0] - source_center[0]
        dy = target_center[1] - source_center[1]

        angle = math.atan2(dy, dx)

        # Start at source circle boundary
        start = point_on_circle(
            source_center,
            source_radius,
            angle
        )

        # End at target circle boundary
        end = point_on_circle(
            target_center,
            target_radius,
            angle + math.pi
        )

        arrow_bbox = draw_arrow(
            image,
            start,
            end
        )

        labels.append(
            (
                1,
                *arrow_bbox
            )
        )

    return image, labels


# ============================================================
# SAVE YOLO LABELS
# ============================================================

def save_labels(label_path, labels):

    with open(label_path, "w") as file:

        for label in labels:

            class_id = label[0]

            x_min = label[1]
            y_min = label[2]
            x_max = label[3]
            y_max = label[4]

            x_center, y_center, width, height = \
                bbox_to_yolo(
                    x_min,
                    y_min,
                    x_max,
                    y_max
                )

            file.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


# ============================================================
# GENERATE DATASET
# ============================================================

print("Generating dataset...")

generated = 0

while generated < NUM_IMAGES:

    image, labels = generate_image()

    if image is None:
        continue

    # Determine split
    random_value = random.random()

    if random_value < TRAIN_RATIO:

        split = "train"

    elif random_value < TRAIN_RATIO + VAL_RATIO:

        split = "val"

    else:

        split = "test"

    # File names
    filename = f"state_machine_{generated:04d}"

    image_path = os.path.join(
        OUTPUT_DIR,
        "images",
        split,
        filename + ".png"
    )

    label_path = os.path.join(
        OUTPUT_DIR,
        "labels",
        split,
        filename + ".txt"
    )

    # Save
    cv2.imwrite(
        image_path,
        image
    )

    save_labels(
        label_path,
        labels
    )

    generated += 1

    if generated % 25 == 0:

        print(
            f"Generated {generated}/{NUM_IMAGES}"
        )


print()
print("Dataset generation complete!")
print()
print("Dataset location:")
print(OUTPUT_DIR)