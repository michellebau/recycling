import os
import glob
import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize

# Set the directory where your images are saved
data_dir = '/Users/michellebautista/Desktop/IS219'

# Specify the image size for your CNN
image_size = (224, 224)

# Specify the mean and std for normalization (standard for ImageNet-pretrained models)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


categories = ['aluminum_cans', 'digital_cameras', 'lead_acid_batteries', 'paper_coffee_cups', 'plastic_utensils']

# Create a list to hold your dataset
dataset = []

# Loop over each category
for i, category in enumerate(categories):
    # Use glob to get the list of image file paths in this category's folder
    image_paths = glob.glob(os.path.join(data_dir, category, '*'))
    
    # Loop over each image path
    for image_path in image_paths:
        # Open the image file
        image = Image.open(os.path.join(data_dir, category, image_path)).convert('RGB')
        
        # Resize the image
        image = Resize(image_size)(image)
        
        # Convert the image to a PyTorch tensor
        image = ToTensor()(image)
        
        # Normalize the tensor
        image = Normalize(mean, std)(image)
        
        # Add the image tensor and its label to the dataset
        dataset.append((image, i))

# Convert the list of samples into a PyTorch tensor
data, targets = zip(*dataset)
data = torch.stack(data)
targets = torch.tensor(targets)

# to confirm everything is ok at this point in preprocessing
print(data.shape)
print(targets.shape)

torch.save(data, 'processed_data.pt')
torch.save(targets, 'processed_targets.pt')