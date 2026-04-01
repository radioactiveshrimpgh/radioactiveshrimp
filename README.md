# To install package with pip:

pip install radioactiveshrimp. 

# To install package with git:

ensure uv is installed on system: curl -LsSf https://astral.sh/uv/install.sh | sh  
clone the repository  
cd radioactiveshrimp  
uv sync  
source .venv/bin/activate  

# Examples: 
## Binary Classification
### To run binary_classification example: 

cd scripts  
python3 binaryclassification_impl.py

### To view result:

binaryclassification_impl.py will save the loss history plot to the current directory as a file named crossentropyloss_YYYYMMDDhhmmss.pdf  
(where YYYYMMDDhhmmss is the time stamp of file creation).  
The fit() function returns the four weight matricies and the loss history so these can be stored as variables and/or printed.   

# HW02Q7

cd scripts  
python3 binaryclassification_animate_impl.py  

This will save output animations for W1-W4 to the scripts/media/ directory. 

# HW02Q8

cd scripts  
./multiclass_impl.sh  
(might require running chmod 775 multiclass_impl.sh to make executable)  
pdf files containing the generated plots for each run of the impl and boxplot should be found in script directory after run is complete.   

# HW03Q6
cd scripts 
./imagenet_impl.sh
may need to run chmod +x to make the script executable. 
Command line arguments for the implemntation script can be modified within the bash script.
A sample train, validation, and testing images as well as pdf files containing the generated plots for each run of the impl should be found in script directory after run is complete.

To perform an inference using the trained model, use the imagenet_inference.py script. A model to load, image to be preicted, and file containing the class names present can be provided as command line arguments but default to those created using the imagenet_impl.sh script.

# HW03Q7
cd scripts
./accnet_impl.sh
may need to run chmod +x to make the script executable. 
Command line arguments for the implemntation script can be modified within the bash script.
Including the --load_file flag will use data stored from a previous run. Not including the flag will load, process, window, and store the data before running the rest of the training, etc. 
A PDF file containing the generated plots for each run of the impl should be found in script directory after run is complete.
