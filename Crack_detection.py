import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Real Image
image = cv2.imread('/content/10-11318_6044-PG_07-2Dgraphic-1.png', cv2.IMREAD_GRAYSCALE)
# Step 1: Enhance Image Contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
enhanced_image = clahe.apply(image)
# Step 2: Thresholding to Create Binary Mask
binary_mask = cv2.inRange(enhanced_image, 100, 255)
# Visualize Binary Mask
plt.figure(figsize=(6, 6))
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask After Thresholding")
plt.show()
# Step 3: Apply Morphological Operations to Remove Noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
# Visualize Cleaned Mask
plt.figure(figsize=(6, 6))
plt.imshow(cleaned_mask, cmap='gray')
plt.title("Cleaned Mask After Morphological Opening")
plt.show()
# Step 4: Connected Component Analysis to Filter Regions by Size
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
# Create Filtered Mask
min_area = 200 # Minimum area threshold for keeping components
filtered_mask = np.zeros_like(cleaned_mask)
for i in range(1, num_labels): # Skip the background
if stats[i, cv2.CC_STAT_AREA] >= min_area:
filtered_mask[labels == i] = 255
# Visualize Filtered Mask
plt.figure(figsize=(6, 6))
plt.imshow(filtered_mask, cmap='gray')
plt.title("Filtered Mask After Removing Small Components")
plt.show()
# Step 5: Dilate Mask to Connect Densely Populated Regions
final_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
# Visualize Final Mask
plt.figure(figsize=(6, 6))
plt.imshow(final_mask, cmap='gray')
plt.title("Final Mask After Dilation")
plt.show()
# Step 6: Apply the Final Mask to the Original Image
output_image = cv2.bitwise_and(enhanced_image, enhanced_image,mask=final_mask)
# Visualize Final Output Image
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(enhanced_image, cmap='gray')
plt.title("Original Enhanced Image")
plt.subplot(1, 3, 2)
plt.imshow(final_mask, cmap='gray')
plt.title("Final Mask Without Splitter Points")
plt.subplot(1, 3, 3)
plt.imshow(output_image, cmap='gray')
plt.title("Final Image Without Splitter Points")
plt.tight_layout()
plt.show()
# Step 7: Crack Detection
edges = cv2.Canny(output_image, 100, 200)
kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_crack)
# Calculate Crack Area
contours, _ = cv2.findContours(refined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
crack_area = sum(cv2.contourArea(contour) for contour in contours)
print("Total Crack Area (in pixels):", crack_area)
# Visualize Detected Cracks
plt.figure(figsize=(6, 6))
plt.imshow(refined_edges, cmap='gray')
plt.title("Detected Cracks")
plt.show()
# Step 8: Delaminated Region Detection (Improved)
substrate_mask = cv2.inRange(output_image, 150, 256) # Adjusted threshold for substrate
coating_mask = cv2.inRange(output_image, 0,180) # Adjusted threshold for coating
black_gap_mask = cv2.inRange(output_image, 0, 80) # Adjusted black gap threshold
# Morphological operations to bridge potential gaps
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_substrate = cv2.dilate(substrate_mask, kernel_dilate, iterations=2)
dilated_coating = cv2.dilate(coating_mask, kernel_dilate, iterations=2)
# Combine masks to isolate delaminated regions
delaminated_mask = cv2.bitwise_and(black_gap_mask,cv2.bitwise_and(dilated_substrate, dilated_coating))
# Post-processing to remove noise and refine delaminated areas
delaminated_mask = cv2.morphologyEx(delaminated_mask, cv2.MORPH_OPEN, kernel_dilate)
delaminated_mask = cv2.morphologyEx(delaminated_mask, cv2.MORPH_CLOSE, kernel_dilate)
# Visualize Detected Delaminated Regions
plt.figure(figsize=(6, 6))
plt.imshow(delaminated_mask, cmap='gray')
plt.title("Improved Detected Delaminated Regions")
plt.show()
# Calculate Delaminated Area
delaminated_contours, _ = cv2.findContours(delaminated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
delaminated_area = sum(cv2.contourArea(contour) for contour in delaminated_contours)
print("Total Delaminated Area (in pixels):", delaminated_area)
# Final Visualization
final_output = cv2.bitwise_or(refined_edges, delaminated_mask)
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(refined_edges, cmap='gray')
plt.title("Cracks")
plt.subplot(1, 3, 2)
plt.imshow(delaminated_mask, cmap='gray')
plt.title("Delaminated Regions")
plt.subplot(1, 3, 3)
plt.imshow(final_output, cmap='gray')
plt.title("Final Detection: Cracks and Delaminated Regions")
plt.tight_layout()
plt.show()