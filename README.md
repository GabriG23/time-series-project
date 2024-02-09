# Discover the World of Time Series
This is the official repository for the Machine Learning in Applications Exam of 24/01/2024. This project aims investigating 4 state-of-the-art methodologies in TAD scenario and compare them with a Bayesian-based one, on a time series dataset collected during several actions performed by a Kuka Robot.

Project Link: [Discover the world of time series](https://github.com/MLinApp-polito/mla-prj-23-fp-01-acg) <br />
Report Link: [Report](https://drive.google.com/file/d/1Ti8NOUoRGIkaq2z8545Hv4CDW0xDBs6c/view?usp=sharing) <br />

### Students

Gabriele Greco - s303435@studenti.polito.it - [github](https://github.com/GabriG23) <br />
Davide Aiello - s303296@studenti.polito.it - [github](https://github.com/davideaiello) <br />
Constantin Clipca - s290214@studenti.polito.it -  [github](https://github.com/PhobosKalDeimos) <br />

### Dataset
The dataset can be found at this [link](https://drive.google.com/file/d/1Fn_KVRpwLedTYU1QgfVCRtkvo1hf_9GB/view?usp=sharing) <br />
It contains five different recording sessions in which the robot performs several actions sampled at different frequencies (1, 10, 100, 200 Hz).

### Testing
For a quick testing you can use the `evaluate.ipynb` notebook by uploading the scores in the `scores_labels` folder.
Each file is a dict containing in `anomaly_scores_norm` index the scores of the model and in `true_labels` index the true labels of the model tested on the corrispective frequency. <br />
You can upload a score by
```
import pickle
with open('model_path', "rb") as file:
      scores = pickle.load(file)
```
## Methodologies

### KMeans
This is classic machine learning distance-based method does not use training data; instead, it focuses on clustering the collision data directly with K-means and computes anomaly scores afterward.
The K-means implementation can be found in the Kmeans folder, which contains a Jupyter notebook.

### HIF
Outlier detection methods are based on isolation trees, where they test the collision data in an unsupervised manner, using only non-anomalous data to build the isolation forest. Subsequently, in a supervised way, known anomalies are added to the forest, and the scores are tested again.
All HIF implementations can be found in the HIF folder, which contains a Jupyter notebook.

### LSTM-AD
This Deep learning technique based on forecasting involve training an LSTM network on non-anomalous data, using it as a predictor over a specified number of time steps. The resulting prediction errors are modeled as a Multivariate Gaussian (MVG) distribution, which is then used to assess the likelihood of test data.
All LSTM-AD code implementations can be found in the LSTM-AD folder.

The program has been executed following these operations:

##### Relevant parser arguments for LSTM-ad and EncDec-AD
- dataset_folder: location of dataset folder
- device: cpu or cuda
- epochs_num: number of epochs
- frequency: frequency of the time series dataset
- model_path: path of the model


##### Operations

Downloading dataset
```
import os, sys
if not os.path.isfile('/content/csv_20220811.zip'):
  !gdown 1P8pCKLI-64_HT91Oqid4RUGtZCUht2c-
  !jar xvf  "/content/csv_20220811.zip"
if not os.path.isdir('/content/csv_20220811'):
  print("Dataset doesn't exist")
```
Train the model
```
!python3 LSTM-AD/train.py --dataset_folder /content/csv_20220811
```
Test the model
```
!python3 LSTM-AD/evaluate.py --dataset_folder /content/csv_20220811  --resume
```

### EncDec-AD
This is a Deep Learning reconstruction-based technique and similar to the previous method, it will train an LSTM network, but this time using an encoder-decoder scheme. This network will learn to reconstruct 'normal' time-series behavior, and we will use the reconstruction error to detect anomalies. All EncDec-AD code implementations can be found in the EncDec-AD folder.

The program has been executed following these operations:

##### Operations

Downloading dataset
```
import os, sys
if not os.path.isfile('/content/csv_20220811.zip'):
  !gdown 1P8pCKLI-64_HT91Oqid4RUGtZCUht2c-
  !jar xvf  "/content/csv_20220811.zip"
if not os.path.isdir('/content/csv_20220811'):
  print("Dataset doesn't exist")
```
Train the model
```
!python3 EncDec-AD/train_embedded.py --dataset_folder dataset/csv_20220811/ --device "cpu" --epochs_num 1 --frequency 1

```
Test the model
```
!python3 EncDec-AD/evaluate.py --dataset_folder dataset/csv_20220811/ --model_path ./checkpoints/best_model.pth --device "cpu" --frequency 1  --resume
```

