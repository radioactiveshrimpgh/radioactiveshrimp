README for Radioactiveshrimp package!

To install package:
pip install radioactiveshrimp. 
You can also find the package content by the same name on github. 

To run binary_classification example: 
cd scripts
python3 binaryclassification_impl.py

To view result:
binaryclassification_impl.py will save the loss history plot to the current directory as a file named crossentropyloss_YYYYMMDDhhmmss.pdf
(where YYYYMMDDhhmmss is the time stamp of file creation).
The fit() function returns the four weight matricies and the loss history so these can be stored as variables and/or printed. 