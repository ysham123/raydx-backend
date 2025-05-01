# Import libraries for file handling, image processing, splitting, numerical operations, and argument parsing
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# Set up argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Preprocess chest X-ray images for pneumonia detection")
parser.add_argument("--input", type=str, required=True, help="Path to the filtered dataset (raydx_dataset/)")
parser.add_argument("--output", type=str, required=True, help="Path to save preprocessed images (processed_dataset/)")

# Parse the arguments
args = parser.parse_args()

# Define input and output paths using the parsed arguments
input_path = args.input
output_path = args.output

#Defining the splits and categories for train/test as well as pneumonia and normal xrays

splits = ["train", "test"]
categories = ["pneumonia", "normal"]

for split in splits:
    for category in categories:
        #this creates the output path
        full_path = os.path.join(output_path, split, category)

        #this creates a directory if it doesn't exist
        os.makedirs(full_path, exist_ok = True)
        

def preprocess_image(input_path, output_path):
    try:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {input_path}")
        
        resized_image = cv2.resize(image, (224,224))
        image_float = np.float32(resized_image)
        normalized_image = image_float / 255.0
        normalized_image = (normalized_image * 255).astype(np.uint8)
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        cv2.imwrite(output_path, normalized_image)
        
        return True
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error has occurred: {e}")
        return False
        

def process_images(input_path, output_path, splits, categories):
    for category in categories:
        # Build the path to the category directory (e.g., raydx_dataset/pneumonia/)
        category_path = os.path.join(input_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory not found: {category_path}")
            continue
        
        # Get list of .jpeg files in the category directory
        jpeg_files = [f for f in os.listdir(category_path) if f.endswith('.jpeg')]
        
        if not jpeg_files:
            print(f"Warning: No .jpeg files found in {category_path}")
            continue
        
        # Split files into train (80%) and test (20%)
        train_files, test_files = train_test_split(jpeg_files, test_size=0.2, random_state=42)
        
        # Process files for each split
        for split, file_list in [("train", train_files), ("test", test_files)]:
            for file in file_list:
                # Build input and output paths
                input_file_path = os.path.join(category_path, file)
                output_file_path = os.path.join(output_path, split, category, file)
                
                # Provide progress feedback
                print(f"Processing {category}/{split}: {file}")
                
                # Preprocess and save the image
                success = preprocess_image(input_file_path, output_file_path)
                if not success:
                    print(f"Failed to process: {input_file_path}")

process_images(input_path, output_path, splits, categories)

def count_processed_images(output_path, splits, categories):
    print("\nProcessed Image Counts:")
    for split in splits:
        for category in categories:
            dir_path = os.path.join(output_path, split, category)
            
            if not os.path.exists(dir_path):
                print(f"Warning: Directory not found: {dir_path}")
                count = 0
            else:
                jpeg_files = [f for f in os.listdir(dir_path) if f.endswith('.jpeg')]
                count = len(jpeg_files)
            
            print(f"{split.capitalize()} {category}: {count} images")

process_images(input_path, output_path, splits, categories)

count_processed_images(output_path, splits, categories)