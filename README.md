# The experiment results and the corresponding code
The datasets and code of the Manuscript THC-241011

## Environment
* python:
* torch:
* torchvision:
* numpy:
* pandas:
  
## Model training
```python
train.py
```

## Core codes
```python
xxxx
```

## Notes
Datasets: 
* SCADI https://archive.ics.uci.edu/dataset/446/scadi 
* Nursery https://archive.ics.uci.edu/dataset/76/nursery
* Maternal Health Risk https://archive.ics.uci.edu/dataset/863/maternal+health+risk
* Heart Disease https://archive.ics.uci.edu/dataset/45/heart+disease
Files:
- The folder SCADI-Dataset contains the children's self-care problems data.
  
- The folder dim-XX contains the extracted XX dimensions of features. Ex. dim-20 represents the folder containing the extracted 20 dimensions of features and presents the corresponding testing result.  Besides, performance_present.py in every dim-XX folder presents the numerical testing result.
  
- data_view.py presents the statistic information. 
  
- ablation_experiment.py is the Python code of the ablation experiment. Besides, it also gives the average accuracy of the proposed method. 
- proposed_model.py is the Python code of the proposed model. 
- neural_network.py is the Python code of the neural network.
- neural_network_with_tripletloss.py is the Python code of the neural network and the tripletloss function is the loss function.
- neural_network_with_focal.py is the Python code of the neural network and the focal method is utilized.
