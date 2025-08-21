import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Define paths
model_path = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\train\Model.pt"
test_images_dir = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\test\test_Data"
output_dir = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_3\outputs"
""


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the trained model
model = YOLO(model_path)

# Perform inference on test images and save results
results = model.predict(source=test_images_dir, save=True, project=output_dir, name="Inference_Results", exist_ok=True)

print(f"Inference Result: {output_dir}/Inference_Results/")