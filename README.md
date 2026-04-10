--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

yolo_finalfinal.ipynb   Main notebook — run top to bottom in order

‘/kaggle/input/datasets/rishabhshenoy03/yolo-merged-dataset-filtered-final/YOLO Merged Dataset Filtered Final’  Dataset zip (upload once to Kaggle)

input/
  datasets/
    rishabhshenoy03/
      yolo-merged-dataset-filtered-final/
        YOLO Merged Dataset Filtered Final/
          labels/
            test/
            train/
          images/
            test/
            train/
      yolo-final-merged/
        YOLO Merged Dataset Filtered/
          labels/
            test/
            train/
          images/
            test/
            train/
  models/
    rishabhshenoy03/
      v111/
        pytorch/
          default/
            1/

==================================================
DATA_ROOT: /kaggle/input/datasets/rishabhshenoy03/yolo-merged-dataset-filtered-final/YOLO Merged Dataset Filtered Final
Project root: /kaggle/working/YOLO_Project
Train images: /kaggle/input/datasets/rishabhshenoy03/yolo-merged-dataset-filtered-final/YOLO Merged Dataset Filtered Final/images/train
Test images : /kaggle/input/datasets/rishabhshenoy03/yolo-merged-dataset-filtered-final/YOLO Merged Dataset Filtered Final/images/test


--------------------------------------------------------------------------------
NOTEBOOK STRUCTURE
--------------------------------------------------------------------------------

Part 0 — Load pretrained model (optional)
Loads a previously saved checkpoint if training is to be resumed or if evaluation is to be performed directly on an existing model.

Part 1 — Setup
Installs and imports the required packages, checks GPU availability, and defines the main constants used throughout the notebook, including image size, batch size, thresholds, and evaluation settings.

Part 2 — Paths and folders
Defines the dataset locations, project root, output folders, and run-specific directories used for checkpoints, plots, logs, and prediction outputs.

Part 3 — Class names and dataset sanity checks
Defines the vehicle class names in the correct label order and verifies that the dataset folders, labels, and file structure are valid before training begins.

Part 4 — Write project source files
Writes the core Python source files for the project into the working directory if they are not already present. These files contain the model architecture, backbone, neck, detection heads, dataset utilities, loss functions, assigner, decoder, and evaluation code.

Part 5 — Import project modules
Adds the project root to the Python path and imports the custom modules created in the previous step so that the notebook can use them as a modular codebase.

Part 6 — Build datasets and dataloaders
Constructs the training, validation, test, and blind-test datasets, applies preprocessing and augmentation, and wraps them in dataloaders for efficient batching.

Part 7 — Visualise training samples with ground-truth boxes
Displays sample training images with annotated bounding boxes to confirm that images and labels have been loaded correctly.

Part 8 — Dataset audit
Summarises dataset composition, class balance, and annotation statistics to identify potential data limitations before training.

Part 9 — Build model
Constructs the full YOLO11-style detector with a ResNet-34 backbone, feature-fusion neck, attention modules, and multi-scale detection heads.

Part 10 — Optimiser, scheduler, assigner, and loss
Initialises the optimiser, learning-rate scheduler, target assigner, and loss functions used during training.

Part 11 — Training and validation functions
Defines the per-epoch training and validation routines, including forward propagation, target assignment, loss computation, optimisation, and metric calculation.

Part 12 — Training
Runs the full training pipeline, including staged backbone freezing, checkpoint saving, metric tracking, and best-model selection.

Part 13 — Plot training curves
Generates and saves the main training curves, including loss components and validation metrics over time.

Part 14 — Load saved checkpoints
Loads the saved best-performing checkpoints for downstream comparison and final evaluation.

Part 15 — Final evaluation table for both checkpoints
Evaluates the saved checkpoints on the test set and summarises their performance using metrics such as mAP, precision, recall, F1, F0.5, and FPS.

Part 16 — Inference speed benchmark
Measures runtime performance in milliseconds per image and frames per second for each saved checkpoint.

Part 17 — Visualise test predictions
Displays predicted and ground-truth bounding boxes on sample test images for qualitative comparison.

Part 18 — Collect full-test predictions
Runs inference over the full test set and stores all predictions, targets, and image identifiers for later analysis.

Part 19 — Failure-case mining
Identifies over-predicted and under-predicted images by comparing prediction counts with ground-truth counts across the test set.

Part 20 — Display worst failure cases
Visualises the most severe failure cases to support qualitative error analysis.

Part 21 — Blind-test analysis summaries
Builds per-class summary tables for the blind-test set, including counts of true positives, false positives, false negatives, AP, precision, recall, and F1.

Part 22 — Save prediction images
Saves all rendered prediction outputs and related metadata to disk for reporting and submission.

Part 23 — Display first 60 prediction images
Displays a gallery of the first 60 saved prediction outputs in the notebook for rapid qualitative review.

Final export cell — Package outputs for download
Exports the run directory, checkpoints, prediction outputs, logs, and summary files into a single package for download or submission.


--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
Core Environment
Python ≥ 3.12
CUDA-enabled GPU recommended (Kaggle T4 x2 used for training)
Required Python Libraries
Deep Learning & Computer Vision
torch ≥ 2.0.0
torchvision ≥ 0.15.0
opencv-python ≥ 4.7.0
albumentations ≥ 1.3.0
Data Handling & Utilities
numpy ≥ 1.24.0
pandas ≥ 1.5.0
Pillow ≥ 9.0.0
Visualisation
matplotlib ≥ 3.7.0
Built-in / Standard Libraries
The following Python standard libraries are used and require no separate installation:
os, sys, math, time, random, json, glob, re, shutil, datetime, pathlib, collections, subprocess, importlib
Custom Project Modules
The notebook writes and imports the following local project modules:
models.modules
models.detection_head
models.backbone_resnet
models.yolo11s
data.dataset
data.transforms
training.assigner
training.loss
training.metrics
Installation (Kaggle)
Most core packages (torch, torchvision, numpy, pandas, matplotlib, Pillow) are preinstalled in the Kaggle environment. If required, missing dependencies can be installed using:
pip install -q albumentations opencv-python
No additional setup is required beyond attaching the dataset and enabling GPU acceleration.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

Upload the dataset files to your Kaggle notebook environment.
Ensure that the image folders, label files, and any required project files are available under the notebook’s working input directories before execution.
Upload or attach the notebook file and open it in Kaggle Notebooks.
In the notebook settings, enable GPU acceleration.
Select a T4 x2 GPU environment if available.
Run the notebook cells from top to bottom in order. Do not skip cells, as earlier sections write the required project .py files, define paths, build datasets, and import the custom modules used later in training and evaluation.
If resuming from a previous run, ensure the required checkpoint files are also attached so that the optional pretrained-model loading section can access them correctly.

Estimated runtimes on Kaggle T4 x2:
Data loading and setup ~2–5 min
Full training (100 epochs) ~1.5 hours
Final evaluation and analysis ~5–10 min
Prediction visualisation/export ~5–10 min


--------------------------------------------------------------------------------
SUMMARY OF FINDINGS
--------------------------------------------------------------------------------

A YOLOv11s detector with ResNet-34 backbone was built 
from scratch and trained on 892 labelled road images from Singapore,
spanning four vehicle classes: Bus, Car, Motorcycle, and Truck.

Final test set performance (60 original field photographs):

  mAP@0.50     : 0.518
  mAP@0.50:0.95: 0.336
  Mean F1      : 0.667
  FPS          :  36.3 (batch size 1)



Key findings:

Adding the P2 detection branch was one of the most important architectural decisions for our model. Because our dataset contains many small and distant vehicles, detecting only from the usual deeper feature maps would have caused fine details to be lost. The P2 branch gave the model access to higher-resolution features, which improved sensitivity to small objects and helped reduce missed detections.
Incorporating C3k2 blocks and the C2PSA attention module made the model much closer to a true YOLO11-style detector rather than a simplified baseline. The C3k2 blocks improved feature fusion across scales, while C2PSA helped the model emphasise informative spatial regions and suppress less useful background information. Together, these modules strengthened localisation and classification performance, especially in cluttered road scenes.
Using a ResNet-34 backbone with staged freezing and unfreezing improved training stability. By freezing the backbone in the earlier phase, we allowed the neck and detection heads to learn the task first before fine-tuning deeper feature extraction layers. This reduced unstable updates early in training and helped the model converge more smoothly than if all layers had been trained aggressively from the start.
Saving and comparing two different best checkpoints showed that model quality cannot be judged by a single metric alone. One checkpoint was selected for the strongest overall mAP, while another was selected under an added precision constraint. This revealed a practical trade-off between overall detection performance and cleaner, more selective predictions, and it made our evaluation more robust than relying on only one “best” model.
The main limitation of the model is still generalisation to unseen data. The notebook includes both standard test evaluation and blind-test analysis, and the gap between these results suggests that the model learns the training and validation distribution better than it generalises to fully unseen scenes. This indicates that further gains are likely to depend less on adding more architecture complexity and more on broader data coverage, stronger variation in training images, and more labelled examples across difficult classes.

================================================================================
