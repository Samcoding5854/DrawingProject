from ultralytics import YOLO
import cv2
from util import draw_circle_with_number,create_class_dataframes,sort_Dimensionss,convert_to_csv,final_processed,replace_pm_with_plus_minus
import pandas as pd
import cv2
import os
import pandas as pd
from paddleocr import PaddleOCR

results = {}

from roboflow import Roboflow
rf = Roboflow(api_key="AgZPub3XxfVEcLiBvqgr")
project = rf.workspace().project("symbols-l8rn4")
model = project.version(2).model


PMdetector = YOLO('small+-ann.pt')

picture = 'data/images/train/drawing2Images_B03145A (2)-3.png'
results = {}
image = cv2.imread(picture)  # Load the image using OpenCV

detections = model.predict(picture, confidence=20, overlap=30).json()
print(detections)
model.predict(picture, confidence=20, overlap=30).save("prediction.jpg")
overlay_image = cv2.imread('pmImage.png')

ocr_model = PaddleOCR(use_angle_cls = True, lang='en')

detections_list = []  # Store all detections for the current image


for prediction in detections['predictions']:
    x1 = float(prediction['x']) - float(prediction['width']) / 2
    x2 = float(prediction['x']) + float(prediction['width']) / 2
    y1 = float(prediction['y']) - float(prediction['height']) / 2
    y2 = float(prediction['y']) + float(prediction['height']) / 2
    class_id = prediction['class_id']
    score = prediction['confidence']
    # box = (x1, y1, x2, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(x1, y1, x2, y2)
    detection_crop = image[y1:y2, x1:x2, :]

    # PMrf = Roboflow(api_key="AgZPub3XxfVEcLiBvqgr")
    # PMproject = PMrf.workspace().project("plus-minus")
    # PMmodel = PMproject.version(3).model

    # # infer on a local image
    # PMdetections = PMmodel.predict(detection_crop, confidence=80, overlap=30).json()

    # PMdetections = PMdetector(detection_crop, conf=0.3)

    # if PMdetections and  PMdetections['predictions']:
    #     for PMdetection in PMdetections['predictions']:
    #         pmx1 = float(PMdetection['x']) - float(PMdetection['width']) / 2
    #         pmx2 = float(PMdetection['x']) + float(PMdetection['width']) / 2
    #         pmy1 = float(PMdetection['y']) - float(PMdetection['height']) / 2
    #         pmy2 = float(PMdetection['y']) + float(PMdetection['height']) / 2       
    #         # pmx1, pmy1, pmx2, pmy2, score, class_id = PMdetection
    #         pmx1, pmy1, pmx2, pmy2 = int(pmx1), int(pmy1), int(pmx2), int(pmy2)
            
    #         # Calculate the dimensions of the bounding box
    #         width = pmx2 - pmx1
    #         height = pmy2 - pmy1
            
    #         imageBB = cv2.rectangle(detection_crop, (pmx1, pmy1), (pmx2, pmy2), (0, 255, 0), 2)

    #         # # Resize the overlay image to match the bounding box dimensions
    #         resized_overlay = cv2.resize(overlay_image, (width, height))

    #         # Overlay the resized image on the original image within the bounding box
    #         detection_crop[pmy1:pmy2, pmx1:pmx2] = resized_overlay
    #         # cv2.imwrite(os.path.join('output_image') + str(pmx1) + '.jpg', detection_crop)
    # else:
    #     detection_crop = detection_crop


    fin_processed = final_processed(detection_crop)
    text_result = ocr_model.ocr(fin_processed)
    print("TEXTING RESULT:", text_result)
    numeric_label = ""
    if text_result is not None:
        for item in text_result:
            numeric_label = ""  # Reset numeric_label for each OCR result
            for sub_item in item:
                label = sub_item[1][0]
                text_score = sub_item[1][1]
                numeric_label += label
                numeric_label = replace_pm_with_plus_minus(numeric_label)


    # if numeric_label != "":
    detection_dict = {'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'text': numeric_label,
                        'bbox_score': score,
                        'text_score': text_score}
    
    detections_list.append(detection_dict)

print("detections list: ",detections_list)
# Store all detections in the results dictionary
results[picture] = {'Detection': detections_list}

print("results: ", results)

class_dataframes = create_class_dataframes(results)

# Display the DataFrames for each class
for class_id, df in class_dataframes.items():
    print(f"Class ID: {class_id}")
    print(df)
    print()


class_dataframes_sorted = sort_Dimensionss(class_dataframes)

# Define the desired class_id order
desired_class_id_order = [5, 8, 3, 1, 11, 9, 10, 4, 17, 6, 13, 16, 2, 14, 15, 7, 12]

# Create a mapping between class_id and DataFrame
class_id_mapping = {}
dataframes_in_order = []

# Combine DataFrames into the list in the desired order
for class_id in desired_class_id_order:
    if class_id in class_dataframes_sorted:
        df = class_dataframes_sorted[class_id]
        df['class_id'] = class_id  # Add 'class_id' column with the current class_id
        dataframes_in_order.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes_in_order)

# Reset the index to ensure consecutive serial numbers
combined_df.reset_index(drop=True, inplace=True)

# Add a new column for the serial number
combined_df['Serial Number'] = range(1, len(combined_df) + 1)

print(combined_df)



Characteristics = ['18', 'Angle', 'Centrality', 'Chamfer', 'Concentricity', 'Diameter', 'Flatness', 'GroupedRoughness', 'Length', 'Parallelity', 'Perpendicularity', 'Radius', 'Roughness', 'Runout', 'Thread', 'Thread-UNC-', 'Total Runout', 'True Position']

output_file = 'output.csv'
convert_to_csv(combined_df, Characteristics, output_file)



center_column_name = "detection_bboxes"  # Replace with the name of the column containing centers
number_column_name = "Serial Number"  # Replace with the name of the column containing numbers


draw_circle_with_number(picture, combined_df, center_column_name, number_column_name,"outputimg.png")


