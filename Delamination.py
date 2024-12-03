import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Real Image
image = cv2.imread('/content/10-11318_6044-PG_07-2Dgraphic-1.png',cv2.IMREAD_GRAYSCALE)
# Step 1: Enhance Image Contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
enhanced_image = clahe.apply(image)
# Step 2: Segment Substrate (Light Grey) and Coating (Dark Grey)
substrate_mask = cv2.inRange(enhanced_image, 150, 200) # Adjust range for light grey (substrate)
coating_mask = cv2.inRange(enhanced_image, 50, 100) # Adjust range for dark grey (coating)
# Step 3: Detect Black Gaps Between Substrate and Coating
black_gap_mask = cv2.inRange(enhanced_image, 0, 100) # Black regions(delaminated gaps)
# Step 4: Focus Only on Black Gaps Near Substrate and Coating
dilated_substrate = cv2.dilate(substrate_mask,cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
dilated_coating = cv2.dilate(coating_mask,cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
# Combine masks to isolate delaminated regions
delaminated_mask = cv2.bitwise_and(black_gap_mask,cv2.bitwise_and(dilated_substrate, dilated_coating))
# Step 5: Contour Detection for Delaminated Regions
contours, _ = cv2.findContours(delaminated_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# Filter Contours by Position and Size (Optional)
height, width = image.shape
filtered_contours = []
for contour in contours:
x, y, w, h = cv2.boundingRect(contour)
# Ensure the contour is near the boundary
if x < 0.1 * width or x + w > 0.2 * width or y < 0.1 * height or y + h > 1 * height:
filtered_contours.append(contour)
# Visualization
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 255), 1) # Delaminated regions in yellow
# Display Results
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title("Delaminated Region Detection (Black Gaps Between Layers)")
plt.show()
# Step 6: Calculate Delaminated Area
delaminated_area_pixels = sum(cv2.contourArea(contour) for contour in filtered_contours)
print("Delaminated Area (in pixels):", delaminated_area_pixels)