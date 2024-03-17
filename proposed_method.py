## Title: A reliable neural network model for self-care problems classification

# Method：
# part A is the neural network to obtain the corresponding embedding of the input. Besides, it is a multi-classification task, the triplet loss can be used. 
# part B is the probabilistic deep neural network to classify the input represented as the corresponding embedding (multinomial distribution)-part B 感觉本质上和A是一样的啊，都是用交叉熵作为loss function

## Contributions: 
# 1. the embedding can have a better representation compared to the discrete representation obtained by inteligent optimiation algorithm, that contributes the classification performance.
# 2. the probabilistic deep neural network makes a robust classification based on the concatenate embeddings, the probability of the classification can be useful to the doctor, be responsible to the patient and be further investigated. 

# The precedure is as follows:
# 1. load the self-care dataset, i.e., SCADI based on ICF-CY
# 2. train the neural network to extract the embeddings of the input data (representing learning)
# 3. concatenate the extracted embeddings and feed it to the probabilistic deep neural network (Bayesian Neural Networks) (supervised learning) to classify the input data 
# 4. calculate the indices of the classification performance (Done)

# （可以尝试对性别参数进行扩增，从而翻倍训练数据。扩充以后 class2 可以达到2个样本，可以使用triplet loss；对于数据不平衡的问题，使用focal loss；最好能原生结合）


## Here start the code

# 1. load the self-care dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import random # the random function
import matplotlib.pyplot as plt # plot like matlab
import umap # dimensionality reduction package
from tensorflow.keras import backend as K # custom loss function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

data_file = './SCADI-Dataset/SCADI.csv'
original_data = pd.read_csv(data_file)

data0 = original_data.loc[original_data['Gender'] == 0]
data0.loc[:, 'Gender'] = 1
data1 = original_data.loc[original_data['Gender'] == 1]
data1.loc[:, 'Gender'] = 0

extend_data = pd.concat([original_data, data0, data1], axis=0) # extend the data by omit the gender

class_name = extend_data['Classes'] # construct the training and text data with input and one-hot label
label_encoder = LabelEncoder()
label_value = label_encoder.fit_transform(class_name)
enc = OneHotEncoder()
one_hot = enc.fit_transform(label_value.reshape(-1, 1))
y_onehot = one_hot.toarray()
x = extend_data.drop(labels=['Classes'], axis=1).values
y = [np.argmax(one_hot) for one_hot in y_onehot]
y = np.asarray(y)
y = y.reshape((-1, 1))
data_set = np.concatenate((x, y), axis=1)  # notice class 2 only has two instances, select the data and construct the training and testing data

# 统计训练数据的类别情况，用于 focal loss 的使用
y_list = [] # 将列表中的列表转换为整形数据，使得可哈希，从而运用 set 函数
for item in y:
    for i in item:
        y_list.append(i)
dict = {} # dict
set = set(y_list) # 去除list中的重复项
for item in set:
    dict.update({item: y_list.count(item)})
print(dict)

# 统一数据集的格式，用于网络的训练，如输入数据 x 和 labe数据 y的格式。其中 x = [[],[],...]; y = [...]。
x_data = x
y_data = np.array(y_list)
print(x_data[0].size)
classes = [0, 1, 2, 3, 4, 5, 6]
x_train = x_data
y_train = y_data
x_test = x_data
y_test = y_data

def data_generator(batch_size=64): # 用于 fit function 从而进行训练
    while True:
        a = [] # list
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(classes, 2) # 从 class 类中随机抽取两个不同的类别
            positive_samples = random.sample(list(x_train[y_train == pos_neg[0]]), 2)
            negative_sample = random.choice(list(x_train[y_train == pos_neg[1]]))
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_sample)
        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

def triplet_loss(y_true, y_pred):
    embedding_size = 100 # the embeding size
    anchor_out = y_pred[:, 0: embedding_size] # embeding size is 100, it can be changed
    positive_out = y_pred[:, embedding_size: embedding_size*2]
    negative_out = y_pred[:, embedding_size*2: embedding_size*3]
    # tensorflow backend function
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)  # l1 dist between anchor <-> positive (it can be changed to l2)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)  # l1 dist between anchor <-> negative

    probs = K.softmax([pos_dist, neg_dist], axis=0) # softmax([pos_dist, neg_dist])
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

# build the triplet loss model
# base model
embedding_size = 100
input_layer = Input((205))
x = Dense(200, activation="relu")(input_layer)
x = Dense(150, activation="relu")(x)
x = Dense(100, activation="relu")(x) # embedding size=100
model = Model(input_layer, x)
model.summary()

# triplet loss model
triplet_model_a = Input((205))
triplet_model_p = Input((205))
triplet_model_n = Input((205))
triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
triplet_model.summary()

# triplet_model.compile(loss=triplet_loss, optimizer="adam")
# triplet_model.fit_generator(data_generator(), steps_per_epoch=150, epochs=50) # 开始训练
# triplet_model.save("triplet_1.h5")

triplet_model = load_model("triplet_1.h5", compile=False)
triplet_model.compile(loss=triplet_loss, optimizer="adam") # 对模型进行编译

model_embedings = triplet_model.layers[-2].predict(x_test, verbose=1) # 得到相应的 embedding
print(model_embedings.shape)

# # reduce the embeddings for visulization
# reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='correlation').fit_transform(model_embedings)
# print(reduced_embeddings.shape)
#
# fig1 = plt.figure(1) # 搭配plt.close() 使用
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=y_test)
# plt.show()
# # plt.draw()
# # plt.pause(6)# 间隔的秒数：6s
# # plt.close(fig1)

# 3. ANN with the focal loss function
# focal loss function, focal_loss
def focal_loss(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2.0
    alpha = tf.constant([[1],[1],[1],[1],[1],[1],[1]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss

# the one-hot label for focal loss function
# y_onehot is the corresponding one hot label
# input data is x_data, the label is y_data
# the embedding is the input data in this situation i.e., model_embedings
# classfication model
input_layer = Input((100)) # embedding size=100
midlle_value = Dense(100, activation="relu")(input_layer)
midlle_value = Dense(50, activation="relu")(midlle_value)
model_output = Dense(7, activation="softmax")(midlle_value) # the model output
class_model = Model(input_layer, model_output)
class_model.summary()

x_train = model_embedings
# y_train = y_data
y_train = y_onehot # for 传统分类

# class_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])
class_model.compile(loss=focal_loss, optimizer="adam", metrics=['accuracy'])
history = class_model.fit(x_train, y_train, epochs=100, batch_size=140, verbose=2, validation_data=(model_embedings, y_train))
print(history.history.keys()) # see the keywords

# see the accuracy
loss = history.history['loss']
acc = history.history['accuracy']

plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, label='Training Acc')
plt.title('Training Acc')
plt.legend()

plt.show()