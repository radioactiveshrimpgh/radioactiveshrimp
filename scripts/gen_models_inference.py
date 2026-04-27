# used to load trained gen model, generate images, and evaluate performance
import argparse
from pathlib import Path
from radioactiveshrimp.deepl import GenModelEval
from torchvision import datasets, transforms
import numpy as np
import cv2

import torch, torchvision
import argparse
import subprocess
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch
from PIL import Image
import zipfile
import io
import torch.optim as optim
from radioactiveshrimp.deepl import VAEModel, VAELoss, GenModelTrainer, CelebAZipDataset
# , GANModel, DiffusionModel,
import onnxruntime as ort

# home_dir = Path.home()
cwd = Path.cwd()

print("begining gen_models_inference.py")
#GET COMMAND LINE ARGS BEFORE STARTING
# take cmd line args ex # epochs, train/vadation dataset ratio
parser = argparse.ArgumentParser(description="A script to load, generate, and evaulate a generative model.")
parser.add_argument("--onnx_name", type=str, default='VAEDecoder_save.onnx', help="saved .ONNX of gen_model to use. default=VAEDecoder_save.onnx")
# parser.add_argument("--onnx_location", type=str, default=cwd, help="directory where onnx file is saved. default=current directory")
parser.add_argument("--debug", type=bool, default=False, help="Boolean to run in debug mode (debug prints). default=False")

args = parser.parse_args()
if args.debug:
    print("running in debug (verbose) mode...")
    print("args have paresed!")
    print(args, '\n')


#import saved model/onnx - to perform inference (test)
ort_session = ort.InferenceSession(args.onnx_name)
latent = np.random.randn(25,128).astype(np.float32)
VAEevaluator = GenModelEval()

#generate 25 images
#pass number of images wanted, will save in function...
print("about to generate 25 samples")
outputs = ort_session.run(None, {'latent': latent})
outputs = outputs[0].reshape(25,3,64,64)
for i in range(len(latent)):
    image = outputs[i]
    image = image.reshape(1,3,64,64)
    image=(image/2+.5).clip(0,1)
    image = (image*255).round().astype(np.uint8)
    image = image[0].transpose(1,2,0)
    img_name = "VAE_images/gen_img_" + str(i+1) +".png"
    Image.fromarray(image).save(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    VAEevaluator.perform_eval_metrics(gray)
    # print("*"*50)
    # print("\n\n")
print("25 samples generated")

# perform evaluation on defined metrics
# for img in VAEImages:
#     img_grey
#     VAEevaluator.perform_eval_metrics(img_grey)

#provide aggregated visualization plot (of metrics)
VAEevaluator.visualize()

print("end of gen_model_inference.py")





