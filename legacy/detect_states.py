import cv2
import numpy as np

# ========================================
# 1. Load image
# ========================================

image = cv2.imread("C:/statetocode/images/test1.png")

if image is None:
    print("Could not load image")
    exit()

print("Image loaded successfully")
print("Size:", image.shape)


# ========================================
# 2. Convert to grayscale
# ========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Slight blur to reduce tiny noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)


# ========================================
# 3. Detect edges
# ========================================

edges = cv2.Canny(
    gray,
    50,
    150
)

cv2.imwrite(
    "C:/statetocode/images/edges.png",
    edges
)


# ========================================
# 4. Detect straight line segments
# ========================================

lines = cv2.HoughLinesP(
    edges,

    # Distance resolution
    rho=1,

    # Angular resolution
    theta=np.pi / 180,

    # Minimum number of votes
    threshold=40,

    # Minimum length of a detected line
    minLineLength=30,

    # Maximum gap between points belonging
    # to the same line
    maxLineGap=15
)


# ========================================
# 5. Check if lines were detected
# ========================================

if lines is None:

    print("No lines detected")
    exit()


print()
print("Detected line segments:", len(lines))
print()


# ========================================
# 6. Draw detected lines
# ========================================

line_image = image.copy()

for i, line in enumerate(lines):

    x1, y1, x2, y2 = line[0]

    # Calculate line length
    length = np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    # Calculate angle
    angle = np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1
        )
    )

    print(
        f"Line {i}: "
        f"({x1}, {y1}) -> ({x2}, {y2}) "
        f"length={length:.1f} "
        f"angle={angle:.1f}°"
    )

    # Draw line
    cv2.line(
        line_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Draw index number
    cv2.putText(
        line_image,
        str(i),
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


# ========================================
# 7. Save result
# ========================================

cv2.imwrite(
    "C:/statetocode/images/detected_lines.png",
    line_image
)

print()
print("Saved detected_lines.png")