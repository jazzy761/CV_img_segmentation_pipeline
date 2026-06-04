### SETUP 

- This document explains how to download, set up, and use the shared Conda environment (xai) and access it inside VS Code.

### Prerequisites

- Ensure you have conda or mini conda installed and shared to vscode terminal to access the environment 

### Setting up Environment 

- Navigate to the environment.yml present in root directory of project 
- conda env create -f environment.yml
- this will create the conda env xai 
- Activate conda environment : conda activate xai 

### environment in vscode

- The conda env is completely upto date with current pytorch and cuda versions. Any code can be ran all paths are aligned 

### Code Commands   

- Calibrations: Its a ipynb so it can be run via notebook ui 
- Model : The model is present in train.py 
- command : python Model/train.py --data Dataset --epochs 10 --batch_size 2 --lr 0.005
- Visualise the results: python Model/visualisation.py
- Running the inference : 
cd D:\Cv_img_segmentation_pipeline
python -m Inference.infer --image "D:\Cv_img_segmentation_pipeline\Dataset\test\WhatsApp Image 2026-06-03 at 3-47-55 PM_jpeg.rf.FdPbK9yTDBb9j7bbYCic.jpeg"

- you can choose any img from the dataset and replace its path 
- Pixel-to-MM Conversion: This is integrated into the inference/infer.py and will run with inference using the above command

