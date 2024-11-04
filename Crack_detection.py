import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and Preprocess the Image
image = cv2.imread('png2pdf-1.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
# Step 1: Define Pixels per Millimeter Conversion
pixels_per_0_25mm = 50
pixels_per_mm = pixels_per_0_25mm / 0.25  # Conversion factor in pixels per mm
conversion_factor = pixels_per_mm ** 2  # Conversion factor for area in pixels^2 to mm^2
# Step 2: Thresholding to Segment the Electrode Area
_, electrode_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
# Calculate Total Electrode Area in Pixels
total_area_pixels = np.sum(electrode_mask == 255)  # Count pixels in electrode area
total_area_mm2 = total_area_pixels / conversion_factor  # Convert to mm²
print("Total Electrode Area (in mm²):", total_area_mm2)
# Step 3: Detect Cracks Within the Coated Area
# Use Canny Edge Detection on the electrode_mask to find cracks
cracks = cv2.Canny(electrode_mask, 50, 150)
# Find contours to quantify cracks
contours, _ = cv2.findContours(cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
crack_area_pixels = sum(cv2.contourArea(contour) for contour in contours)
crack_area_mm2 = crack_area_pixels / conversion_factor  # Convert to mm²
crack_count = len(contours)
# Display crack contours for visualization
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, contours, -1, (0, 0, 255), 1)  # Draw cracks in blue
# Show Detected Cracks for Visual Validation
plt.imshow(output_image)
plt.title("Detected Cracks")
plt.show()
