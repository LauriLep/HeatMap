import pandas as pd
from pdfminer.high_level import extract_pages
from pdf2image import convert_from_path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
import os
from pathlib import Path
from typing import Iterable, Any
import pytesseract
from tqdm import tqdm
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black
import fitz
from reportlab.lib.pagesizes import portrait
from reportlab.lib import fonts
import re
from reportlab.lib.units import inch
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import PyPDF2
from PyPDF2.generic import RectangleObject
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors

# Increase the limit for image size in PIL to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  # Disable the limit

# Path to the input PDF
input_pdf_path = "C:/HeatMap/bottomimages/example.PDF" # put pdf file in here that needs to be scanned
# Path to the output PDF
output_pdf_path = "C:/HeatMap/pdf/example_BW_OCR.pdf"

# Check if the output file already exists
if os.path.exists(output_pdf_path):
    print(f"Output PDF already exists at {output_pdf_path}, skipping processing.")
else:
    # Convert PDF pages to images
    pages = convert_from_path(input_pdf_path, dpi=500)  # Adjust dpi as necessary

    # Process each page image
    processed_images = []
    ocr_texts = []
    for page in pages:
        # Convert page to RGB mode if it's not already
        page = page.convert('RGB')
        # Get the pixel data
        data = page.load()
        # Get the size of the image
        width, height = page.size

        # Process every pixel
        for y in range(height):
            for x in range(width):
                r, g, b = data[x, y]
                if (r, g, b) != (0, 0, 0):  # if not black
                    data[x, y] = (255, 255, 255)  # change to white

        # Save the processed image
        processed_images.append(page)

        # Perform OCR on the processed image
        ocr_result = pytesseract.image_to_data(page, output_type=pytesseract.Output.DATAFRAME)
        ocr_texts.append(ocr_result)

    # Create a new PDF
    c = canvas.Canvas(output_pdf_path)

    # Set font to Arial, size 12
    c.setFont("Helvetica-Bold", 12)  # ReportLab uses "Helvetica" which is similar to Arial

    for img, ocr_df in zip(processed_images, ocr_texts):
        # Get the size of the image
        width, height = img.size
        # Convert the image to a temporary file to draw on canvas
        img_path = "temp_page.png"
        img.save(img_path)
        # Draw the image on the canvas with the original size
        c.setPageSize((width, height))
        c.drawImage(img_path, 0, 0, width, height)

        # Draw the OCR text over the image
        for _, row in ocr_df.iterrows():
            text = row['text']
            if isinstance(text, str) and text.strip():  # Ensure the text is a non-empty string
                c.drawString(row['left'], height - row['top'], text)

        c.showPage()
        os.remove(img_path)

    c.save()

    print(f'Output PDF saved to {output_pdf_path}')



#!!! Step 2: read all texts from PDF file and save texts with x,y- coordinates to dataframe "df_coordinates_texts"


# Initialize an empty list to store the coordinates and texts
data = []

def show_ltitem_hierarchy(o: Any, depth=0):
    """Show location and text of LTItem and all its descendants"""
    if depth == 0:
        print('element                        x1  y1  x2  y2   text')
        print('------------------------------ --- --- --- ---- -----')

    # Append to list if it has bbox and text
    if hasattr(o, 'bbox') and hasattr(o, 'get_text'):
        text = o.get_text().strip()
        bbox = o.bbox
        data.append({
            'Text': text,
            'X1': bbox[0],
            'Y1': bbox[1],
            'X2': bbox[2],
            'Y2': bbox[3]
        })
    
    print(
        f'{get_indented_name(o, depth):<30.30s} '
        f'{get_optional_bbox(o)} '
        f'{get_optional_text(o)}'
    )

    if isinstance(o, Iterable):
        for i in o:
            show_ltitem_hierarchy(i, depth=depth + 1)

def get_indented_name(o: Any, depth: int) -> str:
    """Indented name of LTItem"""
    return '  ' * depth + o.__class__.__name__

def get_optional_bbox(o: Any) -> str:
    """Bounding box of LTItem if available, otherwise empty string"""
    if hasattr(o, 'bbox'):
        return ''.join(f'{i:<4.0f}' for i in o.bbox)
    return ''

def get_optional_text(o: Any) -> str:
    """Text of LTItem if available, otherwise empty string"""
    if hasattr(o, 'get_text'):
        return o.get_text().strip()
    return ''

# Define the path to the PDF file
path = Path("C:/HeatMap/pdf/example_BW_OCR.pdf").expanduser()

# Extract pages from the PDF
pages = extract_pages(path)
show_ltitem_hierarchy(pages)

# Create a DataFrame from the extracted data
df_coordinates_texts = pd.DataFrame(data)

# Save the DataFrame to a CSV file
# df_coordinates_texts.to_csv('ltitem_coordinates_texts.csv', index=False)


#!!! Step 3: modify incorrect values of dataframe "df_coordinates_texts" that OCR have misread, for example "416" = "aie".
# a,A = 4 / i,I,L,l = 1 / e,E,g,G = 6 / O = 0, / T,t = 7 / Q,q = 9
values = {
    "a": [4],
    "A": [4],
    "i": [1],
    "I": [1],
    "L": [1],
    "l": [1],
    "e": [6],
    "E": [6],
    "g": [6],
    "G": [6],
    "O": [0],
    "T": [7],
    "t": [7],
    "Q": [9],
    "q": [9]
}

# Convert dictionary into list of tuples
data = [(key, value[0]) for key, value in values.items()]

# Create DataFrame from list of tuples
df_modify_values = pd.DataFrame(data, columns=['toCorrect', 'rightValue'])

df_coordinates_texts = pd.concat([df_coordinates_texts, df_coordinates_texts]).sort_index(kind='merge')

df_coordinates_texts_split = pd.DataFrame()
df_coordinates_texts_split = df_coordinates_texts['Text'].str.split("",expand=True)

# Function to replace values
def replace_values(cell):
    if cell in df_modify_values['toCorrect'].values:
        index = df_modify_values['toCorrect'].values.tolist().index(cell)
        return str(df_modify_values.loc[index, 'rightValue'])
    return cell

# Apply the function to every cell in the DataFrame
df_coordinates_texts_split = df_coordinates_texts_split.applymap(replace_values)


df_coordinates_texts_split.fillna(np.nan, inplace=True)
df_coordinates_texts_split.replace('', np.nan, inplace=True)

for column in df_coordinates_texts_split.columns:
    df_coordinates_texts_split[column] = df_coordinates_texts_split[column].apply(lambda x: x if not pd.isna(x) else None)


df_coordinates_texts_correct_list = ["".join(str(cell) for cell in row if cell is not None) for _, row in df_coordinates_texts_split.iterrows()]


df_coordinates_texts.reset_index(drop=True, inplace=True)

df_coordinates_texts_correct_list_toDF = pd.DataFrame(df_coordinates_texts_correct_list)
df_coordinates_texts_correct_list_toDF[1] = df_coordinates_texts["Text"]
df_coordinates_texts_correct_list_toDF[2] = ""


for i in range(0, len(df_coordinates_texts_correct_list_toDF[0]), 2):
    df_coordinates_texts_correct_list_toDF.at[i, 2] = df_coordinates_texts_correct_list_toDF.at[i, 0]

for i in range(1, len(df_coordinates_texts_correct_list_toDF[1]), 2):
    df_coordinates_texts_correct_list_toDF.at[i, 2] = df_coordinates_texts_correct_list_toDF.at[i, 1]

df_coordinates_texts["Text"] = df_coordinates_texts_correct_list_toDF[2]

df_coordinates_texts["Text"] = df_coordinates_texts["Text"].str.upper()

df_coordinates_texts.rename(columns={'Text': 'state'}, inplace=True)

#!!! Step 4: create dataframe from data that has meternames (room names), temperatures, co2 values, rh, etc...

path = "C:/HeatMap/data" # insert your data with roomcodes and temperatures
files = [file for file in os.listdir(path) if not file.startswith('.')]
meterData = pd.DataFrame()

time_length1 = len(files)
print("Creating dataframe of meters data")

with tqdm(total=time_length1) as pbar:
    
    for file in files:
        current_data = pd.read_csv(path+"/"+file ,sep= ";" ,  encoding = "UTF-8", keep_default_na=False, dtype='unicode')  
        meterData = pd.concat([meterData,current_data])
    
        pbar.update(1)   
    


index_to_drop = []
for i, row in meterData.iterrows():
    if pd.isna(row.iloc[-1]) or row.iloc[-1] == '' or row.iloc[-1].isspace():
        index_to_drop.append(i)

meterData.drop(index=index_to_drop, inplace=True)

meterData.reset_index(drop=True, inplace=True)

meterData.columns = meterData.iloc[0] 
meterData = meterData[1:]

#!!! step 5: merge data from dataframe meterdata to dataframe df_coordinates_texts and create Merged_information dataframe

# Select only specific columns from meterData before merging
selected_columns = ["ExampleText1",'ExampleText2', 'ExampleText3']  # Columns you want to include

Merged_information = pd.merge(df_coordinates_texts, meterData[selected_columns + ['state']], left_on='state', right_on='state', how='left')

# Drop rows with NaN values in the 'ExampleText1' column
Merged_information.dropna(subset=['ExampleText1'], inplace=True)

filt_ExampleText1 = ["RoomTempState"]

Merged_information = Merged_information[Merged_information["ExampleText1"].isin(filt_ExampleText1)]

Merged_information.drop_duplicates(subset=['state'], inplace=True)

#!!! step 6: add room temperatures from dataframe Merged_information to dataframe df_coordinates_texts 

df_coordinates_texts = pd.merge(df_coordinates_texts, Merged_information[["state","ExampleText1","ExampleText2","ExampleText3"]], left_on='state', right_on='state', how='left')
df_coordinates_texts.dropna(subset=['ExampleText1'], inplace=True)
df_coordinates_texts["Th"] = df_coordinates_texts["ExampleText3"] + df_coordinates_texts["ExampleText2"]
df_coordinates_texts.reset_index(drop=True, inplace=True)
# next replace another room identification string with room temperature
# overwrite every second value of "state" starting from index num 0 with corresponding value of "Th"
for i in range(0, len(df_coordinates_texts["state"]), 2):
    df_coordinates_texts.at[i, "state"] = df_coordinates_texts.at[i, "Th"]

df_coordinates_texts = df_coordinates_texts.drop_duplicates()

df_coordinates_texts.reset_index(drop=True, inplace=True)

for i in range(0, len(df_coordinates_texts["Y1"]), 2):
    df_coordinates_texts.at[i, "Y1"] = df_coordinates_texts.at[i, "Y1"]+80
    

# add colors to temperatures
# average color = green, +-1c average yellow, +-2c purple, +-3c red --> add fading effect between colors  


# Remove commas, replace with dots, and convert to numeric
df_coordinates_texts['ExampleText3_toFloat'] = df_coordinates_texts['ExampleText3'].str.replace(',', '.').astype(float).round(1)

# Calculate the average temperature
th_average = df_coordinates_texts['ExampleText3_toFloat'].mean().round(1)

# Add the average temperature to a new column 'Th_avrg'
df_coordinates_texts['Th_avrg'] = th_average

# Calculate the difference from the average temperature and round to one decimal place
df_coordinates_texts['difference_of_Th_avrg'] = (df_coordinates_texts['ExampleText3_toFloat'] - df_coordinates_texts['Th_avrg']).round(1)

# Calculate the minimum value of the 'ExampleText3_toFloat' column
th_min = df_coordinates_texts['ExampleText3_toFloat'].min()

# Calculate the maximum value of the 'ExampleText3_toFloat' column
th_max = df_coordinates_texts['ExampleText3_toFloat'].max()

# Add the minimum value to a new column 'Th_min'
df_coordinates_texts['Th_min'] = th_min

# Add the maximum value to a new column 'Th_max'
df_coordinates_texts['Th_max'] = th_max

# Define the RGB colors as tuples
blue_rgb = (0, 0, 255)    # min temperature
green_rgb = (0, 255, 0)   # average temperature
red_rgb = (255, 0, 0)     # max temperature


# Function to interpolate between two colors
def interpolate_color(color1, color2, fraction):
    return (
        int(color1[0] + (color2[0] - color1[0]) * fraction),
        int(color1[1] + (color2[1] - color1[1]) * fraction),
        int(color1[2] + (color2[2] - color1[2]) * fraction)
    )

# Function to get the interpolated color based on the temperature value
def get_fade_color(value, th_min, th_max, th_average, blue_rgb, green_rgb, red_rgb):
    if value <= th_average:
        # Interpolate between blue and green
        fraction = (value - th_min) / (th_average - th_min)
        return interpolate_color(blue_rgb, green_rgb, fraction)
    else:
        # Interpolate between green and red
        fraction = (value - th_average) / (th_max - th_average)
        return interpolate_color(green_rgb, red_rgb, fraction)

# Apply the function to each value in 'ExampleText3_toFloat'
df_coordinates_texts['Th_color'] = df_coordinates_texts['ExampleText3_toFloat'].apply(
    lambda x: 'rgb({},{},{})'.format(*get_fade_color(x, th_min, th_max, th_average, blue_rgb, green_rgb, red_rgb))
)


# Create an empty list to store the new rows
new_rows = []

# Iterate over each row of the DataFrame
for index, row in df_coordinates_texts.iterrows():
    # Check if the specific cell value of "state" doesn't include "°C"
    if '°C' not in row['state']:
        # Create a new row by copying the existing row
        new_row = row.copy()
        # Replace the "state" value with the "Th_color" value in the new row
        new_row['state'] = new_row['Th_color']
        # Append the new row to the list of new rows
        new_rows.append(new_row)

# Convert the list of new rows to a DataFrame and concatenate it with the original DataFrame
new_rows_df = pd.DataFrame(new_rows, columns=df_coordinates_texts.columns)
df_coordinates_texts = pd.concat([df_coordinates_texts, new_rows_df], ignore_index=True)

#!!! step 7 open pdf file that has just walls and insert df_coordinates_texts texts to pdf file and print out
# resize pdf picture "only walls" to same size with OCR pdf picture to match the rigth coordinates

# Function to get the size of the first page of the PDF
def get_pdf_page_size(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf = PdfReader(file)
        page = pdf.pages[0]
        width = float(page.mediabox.upper_right[0])
        height = float(page.mediabox.upper_right[1])
        return (width, height)

def resize_pdf(input_pdf, reference_pdf, output_pdf):
    reference_width, reference_height = get_pdf_page_size(reference_pdf)
    
    with open(input_pdf, 'rb') as input_file:
        pdf = PdfReader(input_file)
        writer = PdfWriter()

        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            original_width, original_height = float(page.mediabox.upper_right[0]), float(page.mediabox.upper_right[1])
            scale_x = reference_width / original_width
            scale_y = reference_height / original_height
            
            # Scale the page content
            scale_factor = min(scale_x, scale_y)
            page.scale_by(scale_factor)
            
            # Adjust the media box
            new_mediabox = (0, 0, reference_width, reference_height)
            page.mediabox.lower_left = new_mediabox[:2]
            page.mediabox.upper_right = new_mediabox[2:]
            
            writer.add_page(page)

        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)

input_pdf = "C:/HeatMap/pdf_justwalls/justwalls.pdf" #!!! create this picture that includes only walls of bottomimage
reference_pdf = "C:/HeatMap/pdf/example_BW_OCR.pdf"
output_pdf = "C:/HeatMap/output/heatmap.pdf"

resize_pdf(input_pdf, reference_pdf, output_pdf)

print(f'PDF file "{input_pdf}" has been resized to match the size of "{reference_pdf}" and saved as "{output_pdf}".')


# add dataframe texts to resized "walls" pdf

# Define the function to create a temporary PDF with the text
def create_text_pdf(df, output_path, page_size):
    c = canvas.Canvas(output_path, pagesize=page_size)
    c.setFont("Helvetica", 60)  # Set font and size
    
    # Add color bar to PDF
    bar_x = 50
    bar_y = page_size[1] - 100
    bar_height = 240
    bar_width = 3200
    num_steps = 100
    
    for i in range(num_steps + 1):
        fraction = i / num_steps
        color = get_fade_color(th_min + fraction * (th_max - th_min), th_min, th_max, th_average, blue_rgb, green_rgb, red_rgb)
        c.setFillColorRGB(color[0] / 255, color[1] / 255, color[2] / 255)
        c.rect(bar_x + i * (bar_width / num_steps), bar_y, bar_width / num_steps, bar_height, stroke=0, fill=1)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 60)
    c.drawString(bar_x, bar_y - 100, f'Min: {th_min}')
    c.drawString(bar_x + bar_width / 2, bar_y - 100, f'Average: {th_average}')
    c.drawString(bar_x + bar_width - 250, bar_y - 100, f'Max: {th_max}')
    
    for index, row in df.iterrows():
        text = row['state']
        x1 = row['X1'] #+ 70
        y1 = row['Y1'] #- 130
        
        if 'rgb' in text:
            # Extract the RGB values from the 'state' column
            rgb_values = tuple(map(int, text[4:-1].split(',')))
            normalized_rgb = tuple(val / 255 for val in rgb_values)
            c.setFillColorRGB(*normalized_rgb)  # Set fill color for the ball
            c.circle(x1 + 150, y1+20, 30, stroke=1, fill=1)
        else:
            c.setFillColorRGB(0, 0, 0)  # Set text color to black
            c.drawString(x1, y1, text)
        
    c.save()

# Get the size of the original PDF
input_pdf_path = "C:/HeatMap/output/heatmap.pdf"
original_pdf_size = get_pdf_page_size(input_pdf_path)

# Create a temporary PDF with the texts
temp_pdf_path = "temp_texts.pdf"
create_text_pdf(df_coordinates_texts, temp_pdf_path, original_pdf_size)

# Merge the text PDF with the original PDF
def merge_pdfs(background_pdf_path, overlay_pdf_path, output_pdf_path):
    background = PdfReader(background_pdf_path)
    overlay = PdfReader(overlay_pdf_path)
    writer = PdfWriter()

    for page_num in range(len(background.pages)):
        page = background.pages[page_num]
        overlay_page = overlay.pages[0]  # Use the same overlay page for all pages
        page.merge_page(overlay_page)
        writer.add_page(page)

    with open(output_pdf_path, 'wb') as output_file:
        writer.write(output_file)

# Paths to the input and output PDFs
output_pdf_path = "C:/HeatMap/output/heatmap_final.pdf"

# Merge the PDFs
merge_pdfs(input_pdf_path, temp_pdf_path, output_pdf_path)

print(f'Texts have been added to the PDF and saved as "{output_pdf_path}".')


