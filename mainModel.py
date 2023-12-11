import cv2
from util import draw_circle_with_number,create_class_dataframes,sort_Dimensionss,convert_to_csv,final_processed,OCR_results,convert_PDF_to_Image
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import os
import pandas as pd


results = {}

from roboflow import Roboflow
rf = Roboflow(api_key="AgZPub3XxfVEcLiBvqgr")
project = rf.workspace().project("symbols-l8rn4")
model = project.version(2).model


results = {}

def Balooning_CSV_Generator(pdfPath ,APIChoice):

    # Convert the API choice to lowercase for case-insensitive comparison
    APIChoice = APIChoice.lower()

    # Set the number based on the API choice
    if APIChoice == "yes":
        ocrFlag = 0
    else:
        ocrFlag = 1
    
    convert_PDF_to_Image(pdfPath)

    picture = "output_temp/pdfImage.png"

    image = cv2.imread(picture)  # Load the image using OpenCV

    detections = model.predict(picture, confidence=20, overlap=30).json()
    # print(detections)
    # model.predict(picture, confidence=20, overlap=30).save("prediction.jpg")


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
        # print(x1, y1, x2, y2)
        detection_crop = image[y1:y2, x1:x2, :]
        fin_processed = final_processed(detection_crop)

        filename = f'x1_{x1}.jpg'

        output_path = os.path.join('output_temp', filename)
        cv2.imwrite(output_path, fin_processed)

        if ocrFlag == 0:

            ocr_texts = OCR_results(output_path)
            print("OCR Output:", ocr_texts)

            if ocr_texts:
                fileName = f'{ocr_texts}.jpg'
            else:
                text_result = ocr_model.ocr(output_path)
                # print(text_result)


                if not text_result or not any(text_result[0]):
                    fileName = f'EmptyLabel_{x1}.jpg'
                else:
                    label = text_result[0][0][1][0]
                    dimensions_values = label
                    # print(label)

        elif ocrFlag == 1:        

            text_result = ocr_model.ocr(output_path)
            # print(text_result)


            if not text_result or not any(text_result[0]):
                fileName = f'EmptyLabel_{x1}.jpg'
            else:
                label = text_result[0][0][1][0]
                ocr_texts = label
                # print(ocr_texts)
                # Save the image using the label as the filename

                fileName = f'{ocr_texts}.jpg'

        error_values = ' '.join([text for text in ocr_texts if text.startswith(('+', '-', '±')) and len(text) > 1])
        
        dimensions_values = ' '.join([text[1:] if text.startswith('$') else text for text in ocr_texts if not text.startswith(('+', '-', '±')) or (text.startswith(('+', '-', '±')) and len(text) == 1)])

        # Save the fin_processed image
        output_Path = os.path.join('output_temp', fileName)
        # cv2.imwrite(output_Path, detection_crop)
        # print(f"Processed image saved at: {output_Path}")

        os.remove(output_path)
        # print(f"Original image deleted: {output_path}")

        
        detection_dict = {'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score,
                        'error': error_values,
                        'dimensions': dimensions_values}
        
        detections_list.append(detection_dict)

    # print("detections list: ",detections_list)
    # Store all detections in the results dictionary
    results[picture] = {'Detection': detections_list}

    # print("results: ", results)

    class_dataframes = create_class_dataframes(results)

    # Display the DataFrames for each class
    # for class_id, df in class_dataframes.items():
    #     print(f"Class ID: {class_id}")
    #     print(df)
    #     print()


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

    # print(combined_df)



    Characteristics = ['18', 'Angle', 'Centrality', 'Chamfer', 'Concentricity', 'Diameter', 'Flatness', 'GroupedRoughness', 'Length', 'Parallelity', 'Perpendicularity', 'Radius', 'Roughness', 'Runout', 'Thread', 'Thread-UNC-', 'Total Runout', 'True Position']

    output_file = 'Output.csv'
    convert_to_csv(combined_df, Characteristics, output_file)



    center_column_name = "detection_bboxes"  # Replace with the name of the column containing centers
    number_column_name = "Serial Number"  # Replace with the name of the column containing numbers


    draw_circle_with_number(picture, combined_df, center_column_name, number_column_name,"outputImage.png")

    os.remove(picture)

    return("\n\nThe output image has been stored as 'outputImage.png' and csv file has been saved as 'Output.csv'.")