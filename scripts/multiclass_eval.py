#!/usr/bin/env python

import sys
import getopt
from radioactiveshrimp import deepl
import matplotlib.pyplot as plt
import pandas as p
from datetime import datetime
import os
import glob

def main(argv):   
    try:
        opts, args = getopt.getopt(argv, "hk:", ["keyword="])
        if len(opts) == 0:
           print('Check options by typing:\n{} -h'.format(__file__))
           sys.exit()

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    # print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -k, --keyword\t File Name Keyword')
            sys.exit()
        elif opt in ("-k", "--keyword"):
            keyword = arg
    
    search = "*"+keyword+".csv"
    files = glob.glob(search)
    if not files:
        print(f"No files found matching the pattern: {search}")
    else:   
        dfs=[]
        for file in files:
            df = p.read_csv(file)
            # df['source_file']=os.path.basename(file)
            dfs.append(df)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        all_data = p.concat(dfs, ignore_index=True)
        print(all_data)
        all_data[["train_f1", "train_prec", "train_recall", "train_acc"]].boxplot(ax=axs[0])
        axs[0].set_title("Model Performance Metrics Across Runs - Training") 
        axs[0].set_ylabel("Score") 
        axs[0].grid(axis="y", linestyle="--", alpha=0.5)

        all_data[["test_f1", "test_prec", "test_recall", "test_acc"]].boxplot(ax=axs[1])
        axs[1].set_title("Model Performance Metrics Across Runs - Testing") 
        axs[1].set_ylabel("Score") 
        axs[1].grid(axis="y", linestyle="--", alpha=0.5)
        # plt.show()

        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')
        fileName = date+"_"+keyword+"_boxplot.pdf"
        plt.savefig(fileName)

    print("Complete.")


if __name__ == "__main__":
   main(sys.argv[1:])