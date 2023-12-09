import tempfile
from util import OCR_results
import cv2
from PIL import Image

def set_image_dpi(file_path):

    im = Image.open(file_path)

    length_x, width_y = im.size

    factor = min(1, float(1024.0 / length_x))

    size = int(factor * length_x), int(factor * width_y)

    im_resized = im.resize(size, Image.ANTIALIAS)  # Corrected attribute here

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')

    temp_filename = temp_file.name

    im_resized.save(temp_filename, dpi=(300, 300))

    return temp_filename

# Example usage:
image_path = 'output_images/None_1509.jpg'  # Replace with the path to your image
preprocessed_image = set_image_dpi(image_path)

cv2.imwrite('output_images/None_116951.jpg', preprocessed_image)

# Now you can use 'output_images/None_116951.jpg' for OCR
ocr_texts = OCR_results('output_images/None_116951.jpg')
