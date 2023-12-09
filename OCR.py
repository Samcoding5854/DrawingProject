from PIL import Image
from paddleocr import PaddleOCR
import cv2


ocr_model = PaddleOCR(use_angle_cls = True, lang='en')
image_path = "output_images/None_302.jpg"
img = cv2.imread(image_path)

text_result = ocr_model.ocr(image_path)
print(text_result)


for item in text_result:
        for sub_item in item:
            label = sub_item[1][0]
            print(label)

# img.show()
cv2.waitKey(0)