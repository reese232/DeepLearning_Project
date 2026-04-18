# DeepLearning_Project

A project for the module Theory and Practice of Deep Learning.

This project aims to build a deep learning model that can successfully distinguish spoofs from real audio clips about 70 percent of the time due to the complexity of the task, assuming spoofs to be an anomaly at an occurrence rate of 5%.

Datasets and models saved in the Google Drive:
https://drive.google.com/drive/folders/1NEqy270wGX-fWen06xXojO2GVVNIOPuH?usp=drive_link

---
### Package Dependencies and Imports
```
!pip -q install torch torchaudio scikit-learn pandas matplotlib tqdm optuna


import json
import math
import os
import random
from collections import Counter
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm
```

## Steps to re-train the model
In DL_Model_direct_test_epoch31.ipynb, run all cells until just before Final Testing.
So, sections to run are
- Install dependencies
- Configurations and Data Preprocessing
- Model Architecture
- Training and Evaluation

## Steps to recreate model and performance results
In DL_Model_direct_test_epoch31.ipynb, run the following sections/code
1. Install dependencies
2. Mount Google Drive and set path
3. Imports and Global Configs
4. Dataset and DataLoader
5. Model Architecture
6. Initialise directory where models are saved
7. Helper and Training Functions
8. Final Testing