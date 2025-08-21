import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Input and output folders
INPUT_DIR = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\train\original_dataset_from_kaggle"
OUTPUT_DIR = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\train\dataset"

# Target classes
TARGET_CLASSES = {"car", "motorbike", "bus"}

# Max samples per class
MAX_PER_CLASS = 800

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track counts
class_counts = defaultdict(int)

# Iterate over XML files
for file in os.listdir(INPUT_DIR):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(INPUT_DIR, file)
    base_name = os.path.splitext(file)[0]
    img_file = base_name + ".jpg"

    # Stop if all classes reached limit
    if all(class_counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
        logging.info("All classes reached 50 samples.")
        break

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()

            if cls_name not in TARGET_CLASSES:
                continue

            if class_counts[cls_name] >= MAX_PER_CLASS:
                continue

            # New filenames
            count = class_counts[cls_name] + 1
            out_img_name = f"{cls_name}{count}.jpg"
            out_xml_name = f"{cls_name}{count}.xml"

            out_img_path = os.path.join(OUTPUT_DIR, out_img_name)
            out_xml_path = os.path.join(OUTPUT_DIR, out_xml_name)

            # Copy files if image exists
            img_path = os.path.join(INPUT_DIR, img_file)
            if os.path.exists(img_path):
                shutil.copy(img_path, out_img_path)
                shutil.copy(xml_path, out_xml_path)
                class_counts[cls_name] += 1
                logging.info(f"Copied {img_file} -> {out_img_name} and {file} -> {out_xml_name}")
            else:
                logging.warning(f"Image {img_file} not found for {file}")

            # Stop early if all classes done
            if all(class_counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
                break

    except ET.ParseError:
        logging.error(f"Failed to parse XML: {file}")
        continue
    except Exception as e:
        logging.error(f"Error processing {file}: {str(e)}")
        continue

# Summary
logging.info("Dataset preparation complete.")
for cls in TARGET_CLASSES:
    logging.info(f"{cls}: {class_counts[cls]} samples")
