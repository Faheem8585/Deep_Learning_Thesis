import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load and Preprocess the Image
image = cv2.imread('/content/png2pdf-1.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# Define Scale Conversion
pixels_per_0_25mm = 50
pixels_per_mm = pixels_per_0_25mm / 0.25
conversion_factor = pixels_per_mm ** 2 # Convert from pixels² to mm²
# Step 1: Segment the Substrate and Coating Areas
# Assuming white substrate has higher intensity and coating is gray
_, substrate_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY) #Threshold for white substrate
_, coating_mask = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
# Threshold for gray coating
# Total Electrode Area (in pixels and mm²)
total_area_pixels = np.sum(coating_mask == 255)
total_area_mm2 = total_area_pixels / conversion_factor
# Covered Area (Coating)
covered_area_pixels = np.sum(coating_mask == 255)
covered_area_mm2 = covered_area_pixels / conversion_factor
# Uncovered Area
uncovered_area_mm2 = total_area_mm2 - covered_area_mm2
# Step 2: Crack Detection and Counting
edges = cv2.Canny(coating_mask, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
crack_count = len(contours)
crack_area_pixels = sum(cv2.contourArea(contour) for contour in contours)
crack_area_mm2 = crack_area_pixels / conversion_factor
# Step 3: Delamination Detection
# Use edge detection on substrate boundary to detect gaps
edges_substrate = cv2.Canny(substrate_mask, 50, 150)
# Identify delaminated regions as boundary gaps between substrate and coating
delamination_regions = cv2.bitwise_and(edges_substrate, coating_mask) #Delamination: boundary gaps between layers
# Find contours of delaminated areas
delam_contours, _ = cv2.findContours(delamination_regions,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
delamination_area_pixels = sum(cv2.contourArea(contour) for contour in delam_contours)
delamination_area_mm2 = delamination_area_pixels / conversion_factor
# Step 4: Visualize Results
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, contours, -1, (255, 0, 0), 1) # Draw cracks in red
cv2.drawContours(output_image, delam_contours, -1, (0, 0, 255), 2) # Draw delamination in blue
# Display the output with cracks and delamination highlighted
plt.imshow(output_image)
plt.title("Detected Cracks (RED) and Delamination (BLUE)")
plt.show()
# Print Calculated Results
print("Total Electrode Area (mm²):", total_area_mm2)
print("Covered Area (mm²):", covered_area_mm2)
print("Uncovered Area (mm²):", uncovered_area_mm2)
print("Number of Cracks Detected:", crack_count)
print("Total Crack Area (mm²):", crack_area_mm2)
print("Delamination Area (mm²):", delamination_area_mm2