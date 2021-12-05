# IFES - Applied Computing - AI

This is a framework developed for experiments with bearing faults data sets to evaluate classification models and mitigate similarity bias.
It is part of the dissertation for a Master's degree on Applied Computing on Artificial Intelligence at IFES - Instituto Federal do Espírito Santo - Serra - Brazil.

# Data sets

The following data sets are covered in this framework:

- CWRU (from Case Western Reserve University)
- MFPT (from Mechanical Failures Prevention Group)
- PADERBORN (from the Chair of Design and Drive Technology, Paderborn University, Germany)
- OTTAWA (from the Mechanical Department of the University of Ottawa)

# Requirements

This framework was developed in Python 3.8, using PyCharm IDE.

In order to run this framework, the following packages are required, which are listed in the requirements.txt file:

- patool==1.12
- pip-chill==1.0.1
- pyunpack==0.2.2
- rarfile==4.0
- scikit-learn
- tensorflow
- numpy~=1.19.2
- PyWavelets

Also, for the PADERBORN data set, it is required a compatible unrar software installed to extract the data set files.

# Instructions

Here are the steps to perform experiments using this framework:

- Define if the esperiments will use GPU or not, changing the use_gpu flag.
- Choose which classification models will be evaluated on the clfs list on main().
- Choose the splitting strategy for the experiments on the splits list on main(). Here each data set has different possible splitting strategies.
- Define the number of experiments to be performed on main().
- Define which data set will be used, on the dataset variable on main().

After the experiments execution, the csv files will be available with the results.

In order to view the results again in the future, there is the main_read file. On this file, change the csv file name on metrics.scores on main() to show the results.

# Folders and Files Description

The main framework folders and files are listed below:

- classification_models: Folder with the classification models available for evaluation.
  - auto_cnn.py: CNN model.
  - auto_faultnet.py: FaultNet Model.
  - auto_knn.py: K-nearest neighbors model.
  - auto_lr.py: Linear regressor model.
  - auto_mlp.py: MLP model.
  - auto_random_forest.py: Random forest model.
  - auto_svm.py Support vector machine model.
- datasets: Folder with the data sets available.
  - cwru.py: CWRU data set class definition, with an input of a csv file naming the bearings files names to be used.
  - mfpt.py: MFPT data set class definition.
  - ottawa.py: OTTAWA data set class definition, with an input of downsampling the signal.
  - paderborn.py: PADERBORN data set class definition, with inputs of a csv file naming the bearings and the number of acquisitions to be used.
  - CWRU CSV files: files with the bearings file name and details, with 2 already available (all data set and debug file).
  - PADERBORN CSV files: files with the bearings to be used, with 3 already available (all data set, minimum data for tests and debug file).
- features_extractors: Folder with feature extraction files.
  - heterogeneous.py: Class for statistical and wavelet package features extraction.
  - statisticalfrequency.py: Class with statistical frequency features extraction.
  - statisticaltime.py: Class with statistical time features extraction.
  - wavelet.py: Class with wavelet package features extractors.
- utils: Folder with useful tools.
  - metrics.py: File with function for accuracy and F1 score calculation.
  - persist_results.py: File with functions for saving and loading experiment results on CSV files.
- main.py: File where the experiments settings are defined.
- main_read.py: File to read the experiments results from CSV files.
