import torch
from torchvision import transforms
from PIL import Image
import os
import random

# Paths to your original images
original_images = {
    "flash": [r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\Flash1.png", r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\Flash2.png"],
    "cut": [r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\cuts1.png", r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\cuts2.png"]
}



num_augmented = 25  # 25 per original image
output_folder = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\problem_1\augmented_dataset"
os.makedirs(output_folder, exist_ok=True)

# Augmentation transforms without cutting
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-15, 15)),  # rotate between -15° to +15°
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),   # shift max 10%
        scale=(0.9, 1.1),       # scale ±10%
        shear=10                # shear up to 10°
    )
])

# Generate augmented images
for class_name, img_paths in original_images.items():
    for idx, img_path in enumerate(img_paths, start=1):
        img = Image.open(img_path).convert("RGB")
        for i in range(1, num_augmented + 1):
            augmented_img = augmentation_transforms(img)
            save_name = f"{class_name}{idx}_{i}.png"
            augmented_img.save(os.path.join(output_folder, save_name))

print(f"Augmentation completed! All images saved in {output_folder}")
