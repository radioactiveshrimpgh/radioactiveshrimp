# import matplotlib.pyplot as plt
from radioactiveshrimp import deepl
import polars as pl
import subprocess
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort

import numpy as np
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from datetime import datetime
# from datasets import load_dataset
# from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
import glob
import csv
# import os
# import polars as pl

from pathlib import Path

home_dir = Path.home()

print("begining accnet_impl.py")
#GET COMMAND LINE ARGS BEFORE STARTING
# take cmd line args ex # epochs, train/vadation dataset ratio
parser = argparse.ArgumentParser(description="A script to instantiate and train ACCNet using ACCTrainer")
parser.add_argument("--train_ratio", type=float, default=.1, help="ratio of data to use for training set, default .1 (10%)")
parser.add_argument("--val_ratio", type=float, default=.05, help="ratio of data to use for validation set, default .05 (5%))")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout to be used in ACCNet and classTrainer instances, default .5")
parser.add_argument("--speed_file", type=str, default='decoded_wheel_speed_fl.csv', help="suffix of file name containing speed data")
parser.add_argument("--load_file", action='store_true', help="include flag at run to load stored data - load from file containing the previous preocessed data (saved from previous run)")
parser.add_argument("--data_dir", type=str, default='/data/CPE_487-587/ACCDataset/', help="directory where data files are stored")
parser.add_argument("--status_file", type=str, default='acc_status.csv', help="suffix of acc status file name corresponding to speed data file")
parser.add_argument("-k", type=int, default=10, help="default=10, number of historical values to include per window")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train, default 100")
parser.add_argument("--debug", type=bool, default=False, help="Boolean to run in debug mode (debug prints), default False")


args = parser.parse_args()
if args.debug:
    print("running in debug (verbose) mode...")
    print("args have paresed!")
    print(args, '\n')

# Windowing function
def make_windows(df, k): #merged/zoh dataframe 'result', k window size 
    X_data = df["speed"]
    y_data = df["status"]

    all_X =[]
    all_y=[]
    for i in range(len(X_data),k-1,-1):
        # print("outer: ", i)
        new_row=[]
        for j in range(k+1): #t to t-k ie t-0 to t-10 = 11 total per row
            # print(j)
            new_row.append(X_data[i-1-j])
        all_X.insert(0,new_row)
        all_y.insert(0,y_data[i-1])
    
    # print(all_X)
    # print("*"*50)
    # print(all_y)
    # with open("first_file_out.txt", 'w') as f:
        # f.write(str(all_X))
        # f.write('\n')
        # f.write("*"*50)
        # f.write('\n')
        # f.write(str(all_y))
    # exit()
    return all_X, all_y

#select best GPU
#---------------------------------------------------------------------------------------------------------------
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
    print("best gpu selected...\n")
#-----------------------------------------------------------------------------------------------------------------------------
if args.debug:
    print("grabbing/processing input files/data....")


if args.load_file == False:
    #read in from files
    # print(f"looking for files {args.data_dir}, {args.speed_file}, {args.status_file}")
    files_str = f'{args.data_dir}*{args.speed_file}'
    sfiles_str = f'{args.data_dir}*{args.status_file}'
    # print(files_str, sfiles_str)
    files = sorted(glob.glob(files_str))
    sfiles = sorted(glob.glob(sfiles_str))

    # print(sorted(files), "\n", len(files))
    # print("*"*50)
    # print(sorted(sfiles), "\n", len(sfiles))
    # print("*"*50)

    if len(files) != len(sfiles):
        raise ValueError(f"Number of {args.speed_file} files does not match number of {args.status_file}")

    #---------------------------------------------------------------------------------------------------------------
    # zero order hold, convert units, combine, window
    df_list=[]
    X_windows_list=[]
    y_list=[]

    for i in range(len(files)):
        # print(f'for loop {i}')
        try:
            #ZOH to get similar sample num etc
            #speed data file df:
            speed_file = pl.read_csv(files[i])
            #status file df
            status_file = pl.read_csv(sfiles[i])

            #remove unneeded columns from each (keep time, message)
            speed_file = speed_file.select(["Time", "Message"])
            status_file = status_file.select(["Time", "Message"])

            # status_file.write_csv("inital_acc.csv")

            # print(speed_file)
            # print('*'*50)
            # print(status_file)

            #convert speed_file message column vals to m/s
            speed_file = speed_file.with_columns((pl.col("Message")*(1000./3600.)).alias("speed"))
            speed_file = speed_file.drop("Message")

            # binaize status_file message column 
            status_file = status_file.with_columns(pl.when(pl.col("Message") == 6)
                        .then(1)
                        .otherwise(0)
                        .alias("status"))
            status_file = status_file.drop("Message")

            # print(files[i])
            # print(speed_file, status_file)
            # status_file.write_csv("acc_vals.csv")

            #combine into one frame
            result = speed_file.join_asof(status_file,on="Time",strategy="backward")
            result = result.filter(pl.col("status").is_not_null())
            # print(result)
            # df_list.append(result)


            #create windows
            # make_windows(result,k)
            print(f"calling make_windows on file: {files[i]}")
            X_windows, y = make_windows(result, args.k)
            X_windows_list.append(X_windows)
            y_list.append(y)
            print(f"returned from make_windows on file: {files[i]}")


        except Exception as e:
            print("exception:", e)
            exit()

    # all_vals = pl.concat(df_list)
    # all_vals.write_csv("file_pair_ALL.csv")
    # exit()

    #combine all the windows+label data
    X=[]
    y_labels=[]
    i = 0
    for lists in X_windows_list:
        X += lists
        y_labels += y_list[i]
        i+=1

    np.save('prerun_windowed_data_X.npy', X)
    np.save('prerun_windowed_data_y.npy', y_labels)
    print("saved loaded data. use --load_file to load preprocessed data next run")

else:
    print("loading file of previous run data")
    X = np.load("prerun_windowed_data_X.npy")
    y_labels = np.load("prerun_windowed_data_y.npy")

#-----------------------------------------------------------------------------------------------------------------
if args.debug:
    print("create ACCDataset,train,val,test....")

train_set = deepl.ACCDataset(X, y_labels, args.train_ratio, args.val_ratio, "train", None)
val_set = deepl.ACCDataset(X, y_labels, args.train_ratio, args.val_ratio, "val", train_set.scaler)
test_set = deepl.ACCDataset(X, y_labels, args.train_ratio, args.val_ratio, "test", train_set.scaler)
# print(train_set[0:3])
# print(val_set[0:3])
# print(test_set[0:3])

if args.debug:
    print("acc data sets created.")


#create dataloader
if args.debug:
    print("creating dataloaders...")
    
train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    pin_memory=True,  # Important for faster GPU transfer
    # collate_fn=collate_fn
)

val_loader = DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    pin_memory=True,
)

# initialize ACCNet model
accnet = deepl.ACCNet(drop=args.dropout)
# loss_funct = nn.BCEWithLogitsLoss()
loss_funct = deepl.FocalTverskyLoss()
opt = optim.SGD(accnet.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# opt = optim.Adam(accnet.parameters(), lr=0.001, weight_decay=1e-4)
sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1) # compare to resnet impl idk how will change
accClassTrainer = deepl.ACCTrainer(
    train_loader=train_loader, 
    val_loader=val_loader, 
    # num_classes=num_classes,
    dropout=args.dropout,
    epochs=args.epochs,
    lossfn=loss_funct,
    optimizer=opt,
    scheduler=sch,
    model=accnet)

if args.debug:
    print("all componants intatiated...")
    print("begin accClassTrainer.train()")

# train ACCNet
accClassTrainer.train()
dt = datetime.now()
date = dt.strftime('%Y%m%d%H%M%S')
with open(f"{home_dir}/ACCTrainer_train_progress.txt", "a") as file:
    file.write(f"Training completed at {date}\n" )

if args.debug:
    print("training complete...\nsave model...")

# save as ONNX
accClassTrainer.save()

if args.debug:
    print("model saved...\nrun accClassTrainer.evaluate(), create and save charts")

# evaluate plots - in particular validation accuracy
accClassTrainer.evaluation()

if args.debug:
    print("training, evaluation, complete, charts/figures saved.")

#---------------------------------------------------------------------------   

print("End of accnet_impl.py")

#---------------------------------------------------------------------------
