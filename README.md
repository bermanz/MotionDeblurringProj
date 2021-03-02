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
   1. Download the "train_orig_*" and "val_orig_*" files, and store them under seperate folders (one folder for the training zip files and one for the validation).
1. Generate the dataset:
   ```
   usage: DataSets.py [-h] [-t TRAINING] [-v VALIDATION] [-m {0,1}] [--debug]

   Generate datasets for video-from-image network training.

   optional arguments:
     -h, --help            show this help message and exit
     -t TRAINING, --training TRAINING
                           The full-path to the raw-training data directory
     -v VALIDATION, --validation VALIDATION
                           The full-path to the raw-validation data directory
     -m {0,1}, --masked {0,1}
                           Simulate phase-masking during dataset generation.
                           default: 1 (do simulate)
     --debug               Display an exemplary set of input-targets upon
                           completion
   ``` 
    
## Training
   ```
   usage: train.py [-h] [-e EPOCHS] [-c CHECKPOINT] [-o OUTPUT]

   Train the video-from-image network.
   
   optional arguments:
     -h, --help            show this help message and exit
     -e EPOCHS, --epochs EPOCHS
                           The number of epochs to train for
     -c CHECKPOINT, --checkpoint CHECKPOINT
                           The full-path to a checkpoint for the training to
                           initialize the network with
     -o OUTPUT, --output OUTPUT
                           The full-path to the output directory
   ``` 
## Inference
   ```
   usage: inference.py [-h] [-b BLURRY] [-c CHECKPOINT] [-o OUTPUT]

   Inference a sharp video using the trained video-from-image network.
   
   optional arguments:
     -h, --help            show this help message and exit
     -b BLURRY, --blurry BLURRY
                           The full-path to the blurry input image
     -c CHECKPOINT, --checkpoint CHECKPOINT
                           The full-path to a checkpoint of the trained network
                           with which to inference
     -o OUTPUT, --output OUTPUT
                           The full-path to the output directory
   ``` 
