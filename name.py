import cv2
import numpy as np
import pytesseract
from utils import order_points, four_point_transform

# Load image
image = cv2.imread('document.jpg')
orig = image.copy()

# Resize image
ratio = image.shape[0] / 500.0
image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

# Convert to grayscale and detect edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Find the document contour
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc_cnt = approx
        break

# Apply perspective transform
warped = four_point_transform(orig, doc_cnt.reshape(4, 2) * ratio)

# Convert to black and white
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
scanned = cv2.adaptiveThreshold(
    warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Save result
cv2.imwrite("scanned_output/scanned.jpg", scanned)
print("[INFO] Scanned document saved as 'scanned_output/scanned.jpg'")

# Optional: OCR
text = pytesseract.image_to_string(scanned)
print("[INFO] OCR Result:")
print(text)
