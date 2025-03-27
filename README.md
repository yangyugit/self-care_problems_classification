# Code, Datasets, and Results
The datasets and codes of the Manuscript THC-241011

## Environment
* python: 3.6.13
* tensorflow: 2.0.0
* numpy: 1.19.5
* pandas: 1.1.5
* scikit-learn: 0.24.2
* matplotlib: 3.3.4
  
## Main procedure (in the first level of the "self-care" folder)
1. view the details of the dataset (contains the detailed information of the samples in the SCADI dataset)
```python
data_view.py
```
2. main code (including the training and validation of the proposed neural network)
```python
Example_of_proposed_method_10_fold.py
```
3. performances (presenting the 10-fold performances in 34, 56, and 205 dimensions of the embedding feature)
```python
validation_present.py
```
4. ablation (showing the effects of the embedding neural network (the triplet loss function is utilized) and the classification neural network (the focal loss function is utilized))
```python
ablation_experiment.py
```
5. SHAP analysis (presenting the effect of each feature in the classification result and helping understand the internal logic of the proposed neural network)
```python
SHAP_analysis.py
```

## Core codes
1. loss fucntion formulation
```python
# the triplet_loss function
def triplet_loss(y_true, y_pred):
    embedding_size = 100  # can be changed according to the situation
    anchor_out = y_pred[:, 0: embedding_size]
    positive_out = y_pred[:, embedding_size: embedding_size*2]
    negative_out = y_pred[:, embedding_size*2: embedding_size*3]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)

    probs = K.softmax([pos_dist, neg_dist], axis=0)
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

# the focal loss function
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
2. network structure:
* the embedding neural network
```python
input_layer = Input((205))
x = Dense(200, activation="relu")(input_layer)
x = Dense(150, activation="relu")(x)
x = Dense(embedding_size, activation="relu")(x)
model = Model(input_layer, x)
triplet_model_a = Input((205))
triplet_model_p = Input((205))
triplet_model_n = Input((205))
triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
triplet_model.compile(loss=triplet_loss, optimizer="adam")  # use the triplet loss
```
* the classification neural network
```python
input_layer = Input((embedding_size))
midlle_value = Dense(100, activation="relu")(input_layer)
midlle_value = Dense(50, activation="relu")(midlle_value)
model_output = Dense(7, activation="softmax")(midlle_value)
class_model = Model(input_layer, model_output)
class_model.compile(loss=focal_loss, optimizer="adam", metrics=['accuracy', f1])  # use the focal loss
```

3. hyperparameters
```python
optimizer="adam"
epochs=50
batch_size=32
```

4. augmentation technique(e.g., increasing the dataset by duplicating samples based on gender)
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
1. Dataset: 
* SCADI https://archive.ics.uci.edu/dataset/446/scadi

2. Extend datasets:
* Nursery https://archive.ics.uci.edu/dataset/76/nursery
* Breast Cancer https://archive.ics.uci.edu/dataset/14/breast+cancer

3. Descriptions:
* For the SCADI, we proposed a novel deep learning method for the classification of the children's self-care problems. The corresponding dataset can be found in the "dataset" folder. The example code is included in the first level of the "self-care" folder. Besides, other experimential codes are presented in the "dim-xx" folder, where "xx" means the dimension of the embedding feature.
* For the other similar datasets (i.e., the Nursery and the Breast Cancer), the corresponding datasets can be found in the "dataset" folder. Besides, the corresponding codes are also included in the "nursery" and "breast_cancer" folders, respectively. 
