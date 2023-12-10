from PIL import Image, ImageFilter
from util import OCR_results
import cv2
import numpy as np
from PIL import Image

# Function to resize an image
def resize_image(image_path, target_height=800):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Calculate the target width to maintain the aspect ratio
    aspect_ratio = image.width / image.height
    target_width = int(target_height * aspect_ratio)

    # Resize the image
    resized_image = image.resize((target_width, target_height))

    return resized_image

# Example usage
local_image_path = "output_images/EmptyLabel_1689.jpg"  # Replace with the actual path to your image file
target_height = 200

# Resize the image
resized_image = resize_image(local_image_path, target_height)

# Save or display the resized image
resized_image.save("resized_image.jpg")  # Save the resized image to a file
# resized_image.show()  # Display the resized image

image = cv2.imread('upscaled_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a kernel (structuring element)
kernel = np.ones((12,12), np.uint8)

# Perform erosion
erosion_result = cv2.erode(image, kernel, iterations=1)

# Save the eroded image
cv2.imwrite('eroded_image.jpg', erosion_result)

