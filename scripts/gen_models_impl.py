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

# class CelebAZipDataset(Dataset):
#     def __init__(self, zip_path, transform=None):
#         self.zip_path = zip_path
#         self.transform = transform

#         # Open zip once to collect all image filenames
#         with zipfile.ZipFile(zip_path, 'r') as zf:
#             self.image_names = sorted([
#                 name for name in zf.namelist()
#                 if name.lower().endswith(('.jpg', '.jpeg', '.png'))
#             ])

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self, idx):
#         # Re-open zip per worker (required for DataLoader multiprocessing)
#         with zipfile.ZipFile(self.zip_path, 'r') as zf:
#             with zf.open(self.image_names[idx]) as f:
#                 img = Image.open(io.BytesIO(f.read())).convert('RGB')

#         if self.transform:
#             img = self.transform(img)

#         return img


print("begining gen_models_impl.py")
#GET COMMAND LINE ARGS BEFORE STARTING
# take cmd line args ex # epochs, train/vadation dataset ratio
parser = argparse.ArgumentParser(description="A script to instantiate and train a generative model using GenModelTrainer")
parser.add_argument("--train_ratio", type=float, default=.001, help="ratio of data to use for training set, default .01 (1%)")
parser.add_argument("--x", type=int, default=10, help="save model in training every x epochs, default 10)")
parser.add_argument("--batch_size", type=int, default=128, help="batch size for dataloader, default 128)")
# parser.add_argument("--dropout", type=float, de2fault=0.0, help="dropout to be used in model and classTrainer instances, default 0")
parser.add_argument("--model", type=str, default='VAE', help="default=VAE, type of gen_model to use. options: 'VAE', 'GAN', 'Diffusion'")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train, default 100")
parser.add_argument("--debug", type=bool, default=False, help="Boolean to run in debug mode (debug prints), default False")

args = parser.parse_args()
if args.debug:
    print("running in debug (verbose) mode...")
    print("args have paresed!")
    print(args, '\n')


print('create transform')
if args.model.upper() == 'VAE':
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()
        # transforms.Normalize([0.5]*3, [0.5]*3)  # to [-1, 1]
    ])
else:
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # to [-1, 1]
    ])


print("create dataset")
dataset = CelebAZipDataset(
    zip_path='/data/CPE_487-587/img_align_celeba.zip',
    transform=transform
)

data_len=len(dataset)
train_end = int(data_len*args.train_ratio)
train_set = Subset(dataset, range(0, train_end))

print("create dataloder")
trainloader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


print("iterate dataloader/batch")
batch,_ = next(iter(trainloader))
print(f"Batch shape: {batch.shape}")   # torch.Size([128, 3, 64, 64])
print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")  # [-1.0, 1.0]

if args.debug:
    print("create GenModelTrainer...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#determine desired model and instantiate
if args.model.upper() == 'VAE':
    model=VAEModel(batch_size=args.batch_size, device=device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.optimizer = opt
    loss_funct = VAELoss()
    model.lossfn=loss_funct
    sch = None
    #optimizer Adam - defined in VAEModel
# elif args.model.upper() == 'GAN':
#     GAN(epochs=args.epochs,data=dataloader)
# elif args.model.upper() == 'DIFFUSION':
#     DiffusionModel(epochs=args.epochs,data=dataloader)
else:
    raise ValueError("Model type not recognized! Acceptable models: VAE, GAN, or Diffusion")

#each model needs defined in their init: lossfn, optimizer, scheduler


# instantiate trainer
modelTrainer = GenModelTrainer( 
    device = device,
    epochs=args.epochs,
    lossfn=loss_funct,
    opt=opt,
    sch=sch,
    model=model, 
    saveX = args.x, 
    dataLoader=trainloader)
if args.debug:
    print("all componants intatiated...")
    print("begin GenModelTrainer.train()")


modelTrainer.train() #saves model also

print("GenClassModel has trained and saved.")
print("End of gen_models_impl.py ", args.model)



