import argparse
import onnxruntime as ort
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


#take new image as cmdline arg to predict class
parser = argparse.ArgumentParser(description="A script to load trained CNNTrainer model and make a prediction.")
parser.add_argument("--model", type=str, default='ImageNetCNN.onnx', help=".ONNX saved model file to load, default 'ImageNetCNN.onnx'")
parser.add_argument("--imgfile", type=str, default='imagenet_inference_img.png', help="image file to make prediction on, default 'imagenet_inference_img.png'")
parser.add_argument("--classnames", type=str, default='class_names.txt', help="file containg class names from original data set, default 'class_names.txt'")

args = parser.parse_args()

#load trained model
ort_session = ort.InferenceSession(args.model)

#load image
input_image_path = args.imgfile
input_image = Image.open(input_image_path)

#load classnames
class_names = []
with open(args.classnames, 'r') as file:
    class_names = file.read().split('\n')

# print(class_names)

#make prediciton on image
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images
#     transforms.ToTensor(),  # Convert images to tensors
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
# ])
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = img_transform(input_image).unsqueeze(0).numpy()  # Add batch dimension

# Make predictions
with torch.no_grad():
    # outputs = model(input_tensor)
    # _, predicted_class = torch.max(outputs, 1)
    outputs = ort_session.run(None, {'input': input_tensor})[0]
    predicted_class = np.argmax(outputs, axis=1)
    # print(predicted_class)

predicted_label = class_names[predicted_class.item()]

print(f"Predicted label: {predicted_label}")
