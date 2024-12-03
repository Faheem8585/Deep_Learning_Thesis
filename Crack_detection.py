import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Real Image
image = cv2.imread('/content/10-11318_6044-PG_07-2Dgraphic-1.png',
cv2.IMREAD_GRAYSCALE)
# Step 1: Apply CLAHE to Enhance Contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))
enhanced_image = clahe.apply(image)
# Step 2: Divide Image into Regions (Top-Bottom and Sides)
height, width = enhanced_image.shape
top_bottom = enhanced_image[:height//2, :] # Top and bottom halves
sides = np.hstack((enhanced_image[:, :width//2], enhanced_image[:,width//2:]))
# Step 3: Apply Separate Thresholds for Regions
top_bottom_mask = cv2.inRange(top_bottom, 50, 100) # Adjust range for darker areas
sides_mask = cv2.inRange(sides, 70, 130) # Adjust range for lighter areas on sides
# Combine Masks
full_mask = np.zeros_like(enhanced_image)
full_mask[:height//2, :] = top_bottom_mask
full_mask[:, :width//2] = sides_mask[:, :width//2]
full_mask[:, width//2:] = sides_mask[:, width//2:]
# Step 4: Edge Detection
edges = cv2.Canny(full_mask, 10, 50)
# Step 5: Morphological Operations to Refine Edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 3))
refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# Step 6: Contour Detection
contours, _ = cv2.findContours(refined_edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# Filter Contours by Area
min_area = 10 # Minimum contour area
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
# Visualization
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, filtered_contours, -1, (255, 0, 0), 1) #Cracks in blue
# Display the Refined Detection
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title("Improved Crack Detection (Top-Bottom and Sides)")
plt.show()
# Calculate Crack Area
crack_area_pixels = sum(cv2.contourArea(contour) for contour in filtered_contours)
print("Crack Area (in pixels):", crack_area_pixels)