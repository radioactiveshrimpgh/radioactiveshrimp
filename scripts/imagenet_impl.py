from datasets import load_dataset
from datasets import load_from_disk
import matplotlib.pyplot as plt
from radioactiveshrimp import deepl
import subprocess
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort

print("begining imagenet_impl.py")
#GET COMMAND LINE ARGS BEFORE STARTING
#E. take cmd line args ex # epochs, train/vadation dataset ratio
parser = argparse.ArgumentParser(description="A script to train ImageNetCNN using CNNTrainer")
parser.add_argument("--train_ratio", type=float, default=.001, help="ratio of data to use for training set, default .1 (10%)")
parser.add_argument("--val_ratio", type=float, default=.00025, help="ratio of data to use for validation set, default .05 (5%))")
parser.add_argument("--dropout", type=float, default=.5, help="dropout to be used in ImageNetCNN and classTrainer instances, default .5")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train, default 100")
parser.add_argument("--debug", type=bool, default=False, help="Boolean to run in debug mode (debug prints), default False")


args = parser.parse_args()
if args.debug:
    print("running in debug (verbose) mode...")
    print("args have paresed!")
    print(args)


if args.debug:
    print("parsing ImageNet data")
#A. Read ImageNet data
#fromom update 3
dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

train_dataset = dataset['train']
val_dataset = dataset['validation']

class_names = train_dataset.features['label'].names
num_classes = len(train_dataset.features['label'].names)

with open("class_names.txt", 'w') as file:
    file.write('\n'.join(class_names))


#save one image from the end of validation as the test image for the imagenet_inference.py script
#save the labeled version as well for comparison to result
if args.debug:
    print("selecting test image for later script...")
test_img = dataset['validation'][-1]['image']
test_img_label = dataset['validation'][-1]['label']
test_img.save("imagenet_inference_img.png")

full_label = class_names[test_img_label]
primary_name = full_label.split(',')[0].strip()

plt.figure(figsize=(8, 8))
plt.imshow(test_img)
plt.title(f"ID {test_img_label}: {primary_name}\n({full_label})", fontsize=10)
plt.axis('off')
plt.show()
plt.savefig('imagenet_inference_LABELED.png')

if args.debug:
    print("test image saved as: imagenet_inference_img.png")

# train_size = int(len(dataset['train']) * 0.10) #% 10 percent selection
train_size = int(len(dataset['train']) * args.train_ratio) 
val_size = int(len(dataset['validation']) * args.val_ratio)

train_dataset = dataset['train'].select(range(train_size))
val_dataset = dataset['validation'].select(range(val_size))

if args.debug:
    print("image net data parsed.")
    print("save first train/val image file...")

#-------------------------------------------------------------------
#C. Save example training and validation img in script folder
# save first training set image to current dir
first_example = train_dataset[0]
image = first_example['image']
label_id = first_example['label']

full_label = class_names[label_id]
primary_name = full_label.split(',')[0].strip()

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"ID {label_id}: {primary_name}\n({full_label})", fontsize=10)
plt.axis('off')
plt.show()
plt.savefig('trainingSet_firstImage.png')

#save first validation set image to current dir
first_example = val_dataset[0]
image = first_example['image']
label_id = first_example['label']

full_label = class_names[label_id]
primary_name = full_label.split(',')[0].strip()

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"ID {label_id}: {primary_name}\n({full_label})", fontsize=10)
plt.axis('off')
plt.show()
plt.savefig('validationSet_firstImage.png')

if args.debug:
    print("train/val image files saved")
    print("being transformations...")

#-----------------------------------------------------------------------------
# B. Perform transformations:
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if args.debug:
    print("performed transforms. Now apply transforms....")

# Apply transforms
def preprocess_train(examples):
    # print("proprocess_train running...")
    images = [train_transform(img.convert('RGB')) for img in examples['image']]
    labels = examples['label']
    
    return {
        'pixel_values': images,
        'labels': labels
    }


def preprocess_val(examples):
    # print("proprocess_val running...")

    images = [val_transform(img.convert('RGB')) for img in examples['image']]
    labels = examples['label']
    
    return {
        'pixel_values': images,
        'labels': labels
    }

train_dataset = train_dataset.with_transform(preprocess_train)
val_dataset = val_dataset.with_transform(preprocess_val)

def collate_fn(batch):
    # print("collate_fn running...")
    
    # Extract pixel_values and labels from each item
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }
#--------------------------------------------------------------------
# B. Create DataLoaders
if args.debug:
    print("creating dataloaders...")
    
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    pin_memory=True,  # Important for faster GPU transfer
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    pin_memory=True,
)

if args.debug:
    print("dataloaders created")

def get_best_gpu(strategy="utilization"):
    """
    Select best GPU by 'utilization' or 'memory'.
    """
    # print("get_best_cpu running...")
    if strategy == "memory":
        # Use PyTorch directly for free memory
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i)  # (free, total)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))

if args.debug:
    print("selecting best gpu...")

# Pick strategy: "utilization" or "memory"
device_id = get_best_gpu(strategy="utilization")
device = torch.device(f"cuda:{device_id}")
print(f"Selected GPU: {device_id}")
if args.debug:
    print("best gpu selected...")
    print("instatiate: \nimageNetCNN\nlossfn(loss_funct)\noptimizer(opt)\nscheduler(sch)\ncnnClassTrainer")

#------------------------------------------------------------------------------------
# C. instantiate ImageNetCNN, CNNTrainer classes
#D. use CLE, SGD, LR=0.01, momentum 0.9, wd 1e-4, step lr scheduler stepsize 30 gamma 0.1
imgNetCNN = deepl.ImageNetCNN(num_classes=num_classes, dropout=args.dropout)
loss_funct = nn.CrossEntropyLoss()
opt = optim.SGD(imgNetCNN.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
cnnClassTrainer = deepl.CNNTrainer(
    train_loader=train_loader, 
    val_loader=val_loader, 
    num_classes=num_classes,
    dropout=args.dropout,
    epochs=args.epochs,
    lossfn=loss_funct,
    optimizer=opt,
    scheduler=sch,
    model=imgNetCNN)

# #answers to questions from assignmet:
# # 1. What is the original number of training samples in the dataset?
# print("og # training samples: ", len(dataset['train']))
# # 2. What is the original number of validation samples in the dataset?
# print("og # validaiton samples: ", len(dataset['validation']))
# # 3. How many number of classes are present in the dataset?
# print("number of classes in dataset: ", num_classes)
# # 4. What is the total number of trainable parameters in your CNN model?
# print("total trainable params in CNN model: ", sum(p.numel() for p in imgNetCNN.parameters() if p.requires_grad))
# exit()
# og # training samples:  1281167
# og # validaiton samples:  50000
# number of classes in dataset:  1000
# total trainable params in CNN model:  5464040


if args.debug:
    print("all componants intatiated...")
    print("begin cnnClassTrainer.train()")

cnnClassTrainer.train()

if args.debug:
    print("training complete...\nsave model...")

#--------------------------------------------------------------------------------------
#F. after training, save plot of loss + accruacy vs epochs, commit plot to github
#provide file in PDF

cnnClassTrainer.save()

if args.debug:
    print("model saved...\nrun cnnClassTrainer.evaluate(), create and save charts")

cnnClassTrainer.evaluation()

if args.debug:
    print("training, evaluation, complete, charts/figures saved.")
    
print("End of imagenet_impl.py")


#-----------------------------------------------------------------------------------------------
