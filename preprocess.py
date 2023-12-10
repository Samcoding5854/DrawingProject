import cv2

def upscale_image(input_path, output_path, scale_factor=2):
    # Read the image
    img = cv2.imread(input_path)

    # Check if the image was successfully loaded
    if img is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image using OpenCV's resize function
    height, width = img.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # Save the upscaled image
    cv2.imwrite(output_path, resized_img)
    print(f"Image upscaled and saved to {output_path}")

# Example usage
input_image_path = "output_images/EmptyLabel_1821.jpg"
output_image_path = "upscaled_image.jpg"
upscale_image(input_image_path, output_image_path, scale_factor=65)
