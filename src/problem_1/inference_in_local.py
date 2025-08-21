import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime

# Paths (update these)
model_path = r'C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\defect_detection_model.pth'  # Path to downloaded model
test_images_dir = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\test_dataset"  # Directory with test images
output_dir = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\inference_results"  # Directory for visualizations

# Device (CPU is sufficient)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)  # Background, Cut, Flash
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Get list of images
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    raise FileNotFoundError(f"No images found in {test_images_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process each image
transform = ToTensor()
class_names = ['Background', 'Cut', 'Flash']
confidence_threshold = 0.5

for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_tensor = transform(image).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    # Process predictions
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # Classify and localize
    print(f"\nInference on: {image_path} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    defect_detected = False
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold and label in [1, 2]:  # Cut or Flash
            print(f"Result: Defect Class: {class_names[label]}, Score: {score:.2f}, Box: {box}")
            defect_detected = True
    
    if not defect_detected:
        print("Result: Normal (no defect detected)")
    
    # Visualize results
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if defect_detected:
        for box, label, score in zip(boxes, labels, scores):
            if score >= confidence_threshold and label in [1, 2]:
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1], f"{class_names[label]} {score:.2f}", 
                        bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(10, 10, "Normal (no defect)", bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    
    # Save visualization
    output_path = os.path.join(output_dir, f"result_{image_file}")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to: {output_path}")