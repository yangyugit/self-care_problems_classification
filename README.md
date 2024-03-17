# A deep learning method for self-care problems classification using represent learning and focal loss

Highlights:
- Propose a deep learning method consisting of two sub-deep neural networks for the classification of children’s self-care problems. 
- The first sub-deep neural network utilizes the triplet loss method to compress the dimensions of the feature and preserve the information and the correlation, simultaneously. 
- The second sub-network employs a technology called focal loss to handle the class imbalance problem to improve classification accuracy further. 
- The numerical tests show that the proposed model achieves the state-of-the-art classification performance of the children's self-care problems.

Files:
- The folder SCADI-Dataset contains the children's self-care problems data.
  
- The folder dim-XX contains the extracted XX dimensions of features. Ex. dim-20 represents the folder containing the extracted 20 dimensions of features. 
- data_view.py presents the statistic information. performance_present.py in every dim-XX folder presents the numerical testing result.
  
- ablation_experiment.py is the Python code of the ablation experiment. Besides, it also gives the average accuracy of the proposed method. 
- proposed_model.py is the Python code of the proposed model. 
- neural_network.py is the Python code of the neural network.
- neural_network_with_tripletloss.py is the Python code of the neural network with the tripletloss function.
- neural_network_with_focal.py is the Python code of the neural network with the focal method.

