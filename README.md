# MotionDeblurProj
This repository contains the artifacts related to the my final project, conducted in the deep learning course (05107-25501) in Tel-Aviv University, 2021.

## Video From Blurred Image
Motion blur is a common issue in our reality, as the usage of hand-held cameras with ranging qualities becomes more common. While many techniques attempt to apply straight-forward deblurring of the captured scene, a novel notion struggles to reproduce a sharp video based on a single blurry frame. In this project, I inspected a fusion between the SOTA methods for video-from-image reproduction with a computational-imaging based technique that proved efficient for single-frame deblurring. I demonstrate the qualitative advantage of the proposed fusion over the SOTA blind-deblurring based solutions over a novel dataset with a wide range of real-world scenes.

Examples:
![Alt Text](Examples/1253/comb.gif)
![Alt Text](Examples/515/comb.gif)

## Environment Setup
1. Clone the repository using:
   ```
   git clone --recursive git@github.com:bermanz/MotionDeblurringProj.git
   ```  
1. Download the Raw images required for the dataset generation:
   1. Go to https://seungjunnah.github.io/Datasets/reds.html
   1. Download the "train_orig_*" and "val_orig_*" files, and store them under DataSets/REDS/Raw/train and DataSets/REDS/Raw/val respectively.
1. Generate the dataset by Executing:
   ```
   python DataSets/DataSets.py
   ```    
   
## Training

