from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
import fitz  # PyMuPDF
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import json


def convert_pdf_to_cv2_image_and_save(pdf_path, output_image_path, page_num=0, dpi=300):
    """
    Converts a specific page of a PDF to a cv2 image (numpy array) and saves it as a PNG file.

    Args:
        pdf_path (str): Path to the PDF file.
        output_image_path (str): Path to save the output PNG image.
        page_num (int): Page number to convert (starting from 0).
        dpi (int): DPI (dots per inch) for rendering the PDF.
    """
    # Load the PDF document
    pdf_document = fitz.open(pdf_path)

    # Load the specific page
    page = pdf_document[page_num]

    # Render the page as an image using Pillow
    image = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

    # Convert the PIL image to a cv2 image
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Save the cv2 image as a PNG file
    cv2.imwrite(output_image_path, cv2_image)

    # Close the PDF document
    pdf_document.close()


def create_class_dataframes(results):
    """
    Create DataFrames for each class and store the detections class-wise.

    Args:
        results (dict): Dictionary containing the results.

    Returns:
        dict: A dictionary containing DataFrames for each class.
    """
    class_dataframes = {}

    for image_name, image_data in results.items():
        if 'Detection' in image_data.keys():
            detections = image_data['Detection']
            for detection in detections:
                bbox = detection['bbox']
                class_id = detection['class_id']  # Assuming 'class_id' is present in the 'results' dictionary
                dimension_plate_bbox = '[{} {} {} {}]'.format(bbox[0], bbox[1], bbox[2], bbox[3])
                dimension_plate_bbox_score = detection['bbox_score']
                Dimensions = detection['dimensions']
                errorS = detection['error']
                if class_id not in class_dataframes:
                    class_dataframes[class_id] = []
                
                class_dataframes[class_id].append([image_name, dimension_plate_bbox,
                                                   dimension_plate_bbox_score, Dimensions, errorS])

    for class_id, data in class_dataframes.items():
        df = pd.DataFrame(data, columns=['image_name', 'detection_bboxes',
                                         'dimension_plate_bbox_score', 'Dimensions', 'Error'])
        class_dataframes[class_id] = df

    return class_dataframes

def sort_Dimensionss(class_dataframes):
    """
    Sort the 'Dimensions' column in descending order for each class DataFrame.

    Args:
        class_dataframes (dict): Dictionary containing DataFrames for each class.

    Returns:
        dict: A dictionary containing updated DataFrames for each class with sorted 'Dimensions'.
    """
    for class_id, df in class_dataframes.items():
        df.sort_values(by='Dimensions', ascending=False, inplace=True)

    return class_dataframes




def convert_to_csv(combined_df, Characteristics, output_file):
    """
    Convert the combined DataFrame into a CSV file containing columns: serial_number, Characteristic, Dimensions.

    Args:
        combined_df (pd.DataFrame): The combined DataFrame with class_id, Dimensions, and serial number.
        Characteristics (list): List of class names corresponding to class_id.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    combined_df['Characteristic'] = combined_df['class_id'].map(lambda x: Characteristics[x])
    result_df = combined_df[['Serial Number', 'Characteristic', 'Dimensions','Error']]
    result_df.to_csv(output_file, index=False)



#### FOR BALLOONING ####

def parse_coordinates(coord_str):
    # Helper function to convert a string representation of coordinates to integers
    # Example input: "[525 444 564 483]"
    coord_str = coord_str.strip('[]')
    coords = coord_str.split()
    return list(map(int, coords))

def draw_circle_with_number(image_path, dataframe, center_column, number_column, output_path):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Ensure the image is in RGB mode with 8 bits per channel
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Define a larger font size for the numbers
    font_size = 25  # You can adjust this value to set the font size

    # Load a font (you can specify a different font file if needed)
    font = ImageFont.truetype("arial.ttf", font_size)
    
    # Iterate over each row in the DataFrame
    for index, row in dataframe.iterrows():
        # Extract center and number from the specified columns
        center_x, center_y, _, _ = parse_coordinates(row[center_column])
        number = str(row[number_column])
        # Draw the red circle with the specified center and radius
        circle_radius = 25  # You can adjust this value to set the radius of the circle
        circle_color = (255, 0, 0)  # Red color in RGB format (R, G, B)
        draw.ellipse([center_x-circle_radius, center_y-circle_radius, center_x+circle_radius, center_y+circle_radius], outline=circle_color, width=2)
        # Display the number inside the circle
        text_color = (255, 0, 0)  # Red color in RGB format (R, G, B)
        text_position = (center_x, center_y)  # Convert center coordinates to integers
        draw.text(text_position, number, fill=text_color, anchor="mm", font=font)  # Use the specified font
        
    # Save the image with circles and numbers
    img.save(output_path)

#### Preproccessing for OCR ####

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    return im_bw

def noise_removal(image):
    grayscale(image)
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thick_font(image):
    import numpy as np
    # image = cv2.bitwise_not(image)
    # kernel = np.ones((5,5),np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.bitwise_not(image)
    return (image)

def final_processed(img):
    img_bw = grayscale(img)

    rotated_image = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    no_noise = noise_removal(rotated_image)
    dilated_image = thick_font(no_noise)

    return dilated_image






# def filter_unwanted_characters(numeric_label):
#     # Define the characters you want to filter out
#     unwanted_characters = ['@','$']

#     # Remove unwanted characters from the numeric_label
#     filtered_label = ''.join(char for char in numeric_label if char not in unwanted_characters)

#     return filtered_label


from PIL import Image

def ocr_space_file(filename, overlay=True, api_key='K84842733188957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               "OCREngine": 2,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

def OCR_results(fileName):
    test_file_response = ocr_space_file(filename=fileName)

    test_file_json = json.loads(test_file_response)
    print(test_file_json)
    
    # Check if there are ParsedResults and TextOverlay in the response
    if "ParsedResults" in test_file_json and test_file_json["ParsedResults"]:
        word_texts = []

        for line in test_file_json["ParsedResults"][0].get("TextOverlay", {}).get("Lines", []):
            for word in line.get("Words", []):
                word_texts.append(word.get("WordText", ""))

        # Printing the extracted WordText values
        print(word_texts)
        return word_texts
    else:
        print("Error in OCR processing. Check the response for details.")
        return None
