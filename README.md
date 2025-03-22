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
# the triplet_loss function
def triplet_loss(y_true, y_pred):
    embedding_size = 100
    anchor_out = y_pred[:, 0: embedding_size] 
    positive_out = y_pred[:, embedding_size: embedding_size*2]
    negative_out = y_pred[:, embedding_size*2: embedding_size*3]
    
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)  
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)  

    probs = K.softmax([pos_dist, neg_dist], axis=0) # softmax([pos_dist, neg_dist])
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

def focal_loss(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2.0
    alpha = tf.cast(alpha_0, tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss
```
network structure:
1. the embedding neural network
```python
embedding_size = 100
input_layer = Input((205))
x = Dense(200, activation="relu")(input_layer)
x = Dense(150, activation="relu")(x)
x = Dense(100, activation="relu")(x)
model = Model(input_layer, x)
triplet_model_a = Input((205))
triplet_model_p = Input((205))
triplet_model_n = Input((205))
triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
```
2. the classification neural network
```python
input_layer = Input((embedding_size)) # embedding size=100
midlle_value = Dense(100, activation="relu")(input_layer)
midlle_value = Dense(50, activation="relu")(midlle_value)
model_output = Dense(7, activation="softmax")(midlle_value)
class_model = Model(input_layer, model_output)
```

hyperparameters
```python
optimizer="adam"
epochs=50
batch_size=32
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
