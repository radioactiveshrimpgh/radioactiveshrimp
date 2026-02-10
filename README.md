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

