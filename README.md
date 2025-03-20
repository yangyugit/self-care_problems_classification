# The experiment results and the corresponding code

## Model training
'''python3 main.py'''

Highlights and Contributions can be seen in the corresponding manuscript (which has been submitted to AEJ). 

Files:
- The folder SCADI-Dataset contains the children's self-care problems data.
  
- The folder dim-XX contains the extracted XX dimensions of features. Ex. dim-20 represents the folder containing the extracted 20 dimensions of features and presents the corresponding testing result.  Besides, performance_present.py in every dim-XX folder presents the numerical testing result.
  
- data_view.py presents the statistic information. 
  
- ablation_experiment.py is the Python code of the ablation experiment. Besides, it also gives the average accuracy of the proposed method. 
- proposed_model.py is the Python code of the proposed model. 
- neural_network.py is the Python code of the neural network.
- neural_network_with_tripletloss.py is the Python code of the neural network and the tripletloss function is the loss function.
- neural_network_with_focal.py is the Python code of the neural network and the focal method is utilized.
