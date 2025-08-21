import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# Input directory (adjust as needed)
INPUT_DIR = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\train\dataset"

# Target classes
TARGET_CLASSES = {"car", "motorbike", "bus"}

# Initialize data structures
class_counts = defaultdict(int)
areas = defaultdict(list)
widths = defaultdict(list)
heights = defaultdict(list)

# Parse XML files
for file in os.listdir(INPUT_DIR):
    if file.endswith(".xml"):
        xml_path = os.path.join(INPUT_DIR, file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                cls_name = obj.find("name").text.strip().lower()
                if cls_name in TARGET_CLASSES:
                    class_counts[cls_name] += 1
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height
                    areas[cls_name].append(area)
                    widths[cls_name].append(width)
                    heights[cls_name].append(height)
        except ET.ParseError:
            print(f"Error parsing {file}")
            continue

# Plot 1: Number of objects per class
plt.figure(figsize=(8, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title("Number of Objects per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("class_distribution.png")
plt.close()

# Plot 2: Box plot of bounding box areas
plt.figure(figsize=(8, 6))
data = [areas[cls] for cls in TARGET_CLASSES]
plt.boxplot(data, labels=TARGET_CLASSES)
plt.title("Bounding Box Area Distribution per Class")
plt.xlabel("Class")
plt.ylabel("Area (pixels²)")
plt.savefig("area_distribution.png")
plt.close()

# Plot 3: Box plot of bounding box widths
plt.figure(figsize=(8, 6))
data = [widths[cls] for cls in TARGET_CLASSES]
plt.boxplot(data, labels=TARGET_CLASSES)
plt.title("Bounding Box Width Distribution per Class")
plt.xlabel("Class")
plt.ylabel("Width (pixels)")
plt.savefig("width_distribution.png")
plt.close()

# Plot 4: Box plot of bounding box heights
plt.figure(figsize=(8, 6))
data = [heights[cls] for cls in TARGET_CLASSES]
plt.boxplot(data, labels=TARGET_CLASSES)
plt.title("Bounding Box Height Distribution per Class")
plt.xlabel("Class")
plt.ylabel("Height (pixels)")
plt.savefig("height_distribution.png")
plt.close()

print("Visualizations saved: class_distribution.png, area_distribution.png, width_distribution.png, height_distribution.png")