# Code, Datasets, and Results
The datasets and code of the Manuscript THC-241011

## Environment
* python: 3.6
* tensorflow: 2.0.0
* keras-applications: 1.0.8
* keras-preprocessing: 1.1.2
* numpy: 1.19.2
* pandas: 0.20.3
* scikit-learn: 0.24.2
* matplotlib: 3.3.4
  
## Model training
```python
train.py
```

## Core codes
loss fucntion formulation
```python
xxxx
```
network structure
```python
xxxx
```
hyperparameters
```python
xxxx
```
augmentation technique(e.g., increasing the dataset by duplicating samples based on gender)
```python
data_file = '../dataset/SCADI.csv'
original_data = pd.read_csv(data_file)

data0 = original_data.loc[original_data['Gender'] == 0]
data0.loc[:, 'Gender'] = 1
data1 = original_data.loc[original_data['Gender'] == 1]
data1.loc[:, 'Gender'] = 0
extend_data = pd.concat([original_data, data0, data1], axis=0)
```

## Notes
Dataset: 
* SCADI https://archive.ics.uci.edu/dataset/446/scadi

Extend:
* Nursery https://archive.ics.uci.edu/dataset/76/nursery
* Maternal Health Risk https://archive.ics.uci.edu/dataset/863/maternal+health+risk
* Heart Disease https://archive.ics.uci.edu/dataset/45/heart+disease

Descriptions:
* The folder dim-XX contains the extracted XX dimensions of features. Ex. dim-20 represents the folder containing the extracted 20 dimensions of features and presents the corresponding testing result.  Besides, performance_present.py in every dim-XX folder presents the numerical testing result.
* data_view.py presents the statistic information. 
* ablation_experiment.py is the Python code of the ablation experiment. Besides, it also gives the average accuracy of the proposed method. 
* proposed_model.py is the Python code of the proposed model. 
* neural_network.py is the Python code of the neural network.
* neural_network_with_tripletloss.py is the Python code of the neural network and the tripletloss function is the loss function.
* neural_network_with_focal.py is the Python code of the neural network and the focal method is utilized.
