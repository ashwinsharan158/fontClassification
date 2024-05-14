import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import sys
import warnings


def split_image_into_patches(image, patch_size):
    """
    Split an image into patches of 128 X 128 with a stride of 28 pixels. 

    Args:
        image (PIL.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: List of image patches.
    """"""
    Split an image into patches.

    Args:
        image (PIL.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: List of image patches.
    """
    width, height = image.size

    # Pad the image if smaller than 128x128
    if width < 128 or height < 128:
        pad_width = max(0, 128 - width)
        pad_height = max(0, 128 - height)
        padding = (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2)
        image = ImageOps.expand(image, padding, fill='white')
    
    # Ensure RGB image
    image = image.convert('RGB')
    # Divide into patches
    stride = 28
    patch_size = 128
    patches = []
    width, height = image.size
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches

def get_majority_prediction(predictions):
    """
    Get the majority prediction from a list of predictions.

    Args:
        predictions (list): List of prediction values.

    Returns:
        int: Majority prediction.
    """
    unique, counts = np.unique(predictions, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    return majority_label

# Define transformation for resizing and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_font_of_image(image_path):
    """
    Get the font of an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Font of the image.
    """
    try:
        patch_predictions = []
        pil_image = Image.open(image_path)
        patches = split_image_into_patches(pil_image, patch_size=128)
        for patch in patches:
            patch = transform(patch).unsqueeze(0)
            if torch.cuda.is_available():
                patch = patch.cuda()
            else:
                patch = patch.cpu()
            outputs = resnet(patch)
            _, predicted = torch.max(outputs, 1)
            patch_predictions.append(predicted.item())
        majority_prediction = get_majority_prediction(patch_predictions)
        return font_class[majority_prediction]
    except Exception as e:
        return str(e)

def main():
    """
    Main function to get font of the image specified in the command line argument.
    """
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        font = get_font_of_image(image_path)
        print("Font of the image:", font)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Warnings 
    warnings.filterwarnings('ignore')
    # Load pre-trained ResNet model
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Modify the last layer for your specific number of classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),  # Add a new fully connected layer
        nn.ReLU(),  # Add ReLU activation function
        nn.Dropout(p=0.5),  # Add Dropout with a dropout probability of 0.5
        nn.Linear(512, 10)  # Output layer
    )
    if torch.cuda.is_available():
        resnet.cuda()
    model_path = 'final_font_class_10_model.pth'
    resnet.load_state_dict(torch.load(model_path))
    resnet.eval()

    font_class = {0: 'AguafinaScript', 1: 'AlexBrush', 2: 'Allura', 3: 'Canterbury', 4: 'GreatVibes',
                  5: 'Holligate Signature', 6: 'I Love Glitter', 7: 'James Fajardo', 8: 'OpenSans', 9: 'alsscrp'}

    main()
