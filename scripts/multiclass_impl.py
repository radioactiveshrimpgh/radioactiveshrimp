#!/usr/bin/env python3

import sys
import getopt
from radioactiveshrimp import deepl as d
import pandas as p
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import csv
from datetime import datetime
import torch.optim as optim
import torch.nn as nn


def main(argv):
    #default values
    file_loc = '../data/Android_Malware.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eta = 0.00001
    epochs = 100000
    loss_fn = nn.CrossEntropyLoss()
    # model = d.SimpleNN().to(device)
    optimizer = None
    keyword = 'hw02'
      
    try:
        opts, args = getopt.getopt(argv, "hf:r:e:l:o:d:k:", ["file_loc=", "learning_rate=", "epochs=","loss_fn=","optimizer=",'device=','keyword='])
        #if len(opts) == 0:
        #    print('Check options by typing:\n{} -h'.format(__file__))
        #    sys.exit()

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -f, --file_loc\t Location of data file')
            print('\t -r, --learning_rate\t Learning Rate (eta)')
            print('\t -e, --epochs\t\t Max Epochs')
            print('\t -l, --loss_fn\t\t Loss function for training')
            print('\t -o, --optimizer\t\t Optimizer ')
            print('\t -d, --device\t\t Device (GPU?)')
            print('\t -k, --keyword\t\t Keyword for appending to save file name')
            sys.exit()
        elif opt in ("-f", "--file_loc"):
            file_loc = arg
        elif opt in ("-r", "--learning_rate"):
            eta = arg
        elif opt in ("-e", "--epochs"):
            epochs = arg
        elif opt in ("-l", "--loss_fn"):
            loss = arg
        elif opt in ("-o", "--optimizer"):
            optimizer = arg
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-k", "--keyword"):
            keyword = arg
    #-------------------------------------------------------------------------------
    df = p.read_csv(file_loc)

    # filter out bad rows
    mask = df.isin(["BENIGN"])
    df = df[~mask.any(axis=1)]
    mask = df.isin(["SCAREWARE"])
    df = df[~mask.any(axis=1)]

    #preprocess data
    y = df['Label'].to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    class_labels = np.unique(y)


    # print(df.columns)
    df = df.drop('Label',axis=1)
    df = df.drop('Flow ID', axis=1)
    df=df.drop(' Source IP',axis=1)
    df=df.drop(" Source Port", axis=1)
    df=df.drop(" Destination IP", axis=1)
    df=df.drop(" Destination Port", axis=1)
    df=df.drop(" Protocol", axis=1)
    df=df.drop(" Timestamp", axis=1)

    X = df.to_numpy(dtype=np.float32)
    # print(X) 
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    #create model
    m = d.ClassTrainer(X_train, y_train, 4)

    if len(opts)>0:
        m.eta = eta
        m.epochs = epochs
        m.loss = loss_fn
        m.device = device
        if optimizer:
            m.optimizer = optimizer

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=m.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=m.device)

    print("begining training...")
    m.train()
    print("training done!")

    print("begining test:")
    m.test(X_test_tensor, y_test_tensor)
    print("testing done")

    # print(X[2].shape)
    # pred_tensor = torch.as_tensor((X[:6]), dtype=torch.float32, device=m.device)
    # print(pred_tensor)
    # print("starting prediction:")
    # pred = m.predict(pred_tensor)
    # print("prediction:", pred)
    m.classes_ = class_labels
    train_f1,train_prec,train_recall,train_acc,test_f1,test_prec,test_recall,test_acc=m.evaluation(m.loss_vector, m.accuracy_vector)


    dt = datetime.now()
    date = dt.strftime('%Y%m%d%H%M%S')
    fileName = date+'_'+keyword+'.csv'
    print(fileName)
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['train_f1','train_prec','train_recall','train_acc','test_f1','test_prec','test_recall','test_acc'])
        writer.writerow([train_f1,train_prec,train_recall,train_acc,test_f1,test_prec,test_recall,test_acc])

    print("Complete.")



if __name__ == "__main__":
   main(sys.argv[1:])


#-----------------------------------------------------------------------
