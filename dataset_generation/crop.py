#code from chatgpt
import cv2
import numpy as np

# Load the image
image_path = 'notes_a-g-1-sample/A4.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to get a binary image
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the image using the bounding box
cropped_image = image[y:y+h, x:x+w]

# Save the cropped image
output_path = 'notes_a-g-1-sample/cropped_image.png'
cv2.imwrite(output_path, cropped_image)

print(f'Cropped image saved to {output_path}')