# 1. load the self-care dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import random # the random function
import matplotlib.pyplot as plt # plot like matlab
# import umap # dimensionality reduction package
from tensorflow.keras import backend as K # custom loss function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle

# 2. load the self-caring data
data_file = '../SCADI-Dataset/SCADI.csv'
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
data_set = np.concatenate((x, y), axis=1)  # notice class 2 only has two instances

y_list = [] # 将列表中的列表转换为整形数据，使得可哈希，从而运用 set 函数
for item in y:
    for i in item:
        y_list.append(i)

dict = {} # dict
set = set(y_list) # 去除list中的重复项
for item in set:
    dict.update({item: y_list.count(item)})

# the alpha in focal loss function
list_base = [2, 2, 2, 2, 2, 2, 2]
list_all = list(dict.values())
global alpha_0
alpha_0 = (np.array(list_all) - np.array(list_base))/sum(np.array(list_all) - np.array(list_base)) + 1
alpha_0.shape = 7,1

x_data = x # the input measurements
y_data = np.array(y_list) # the int label
y_onehot # the one-hot label

# prepare the k-fold data
# select the base data of every kind, that every kind contains at least 2 instances
index0 = [(random.sample(list(np.where(y_data==0)[0]), 2))]
index1 = [(random.sample(list(np.where(y_data==1)[0]), 2))]
index2 = [(random.sample(list(np.where(y_data==2)[0]), 2))]
index3 = [(random.sample(list(np.where(y_data==3)[0]), 2))]
index4 = [(random.sample(list(np.where(y_data==4)[0]), 2))]
index5 = [(random.sample(list(np.where(y_data==5)[0]), 2))]
index6 = [(random.sample(list(np.where(y_data==6)[0]), 2))]
index_0 = np.concatenate((index0,index1,index2,index3,index4,index5,index6), axis=1)[0]
x_index0 = x_data[index_0]
y_index0 = y_data[index_0]
y_onehot_index0 = y_onehot[index_0]

x_left = np.delete(x_data, index_0, axis=0)
y_left = np.delete(y_data, index_0, axis=0)
y_onehot_left = np.delete(y_onehot, index_0, axis=0)

# prepare the data for triplet loss function
def data_generator(batch_size=64): # 用于 fit function 从而进行训练
    classes = [0, 1, 2, 3, 4, 5, 6]
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

def triplet_loss(y_true, y_pred, embedding_size = 70):
    anchor_out = y_pred[:, 0: embedding_size] # embeding size is 100, it can be changed
    positive_out = y_pred[:, embedding_size: embedding_size*2]
    negative_out = y_pred[:, embedding_size*2: embedding_size*3]
    # tensorflow backend function
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)  # l1 dist between anchor <-> positive (it can be changed to l2)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)  # l1 dist between anchor <-> negative

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

# 自定义评价函数
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def recall_score(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_score(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 3. ten-fold testing
val_acc_list = []; val_f1_list = []; # record the performance
val_recall_list = []; val_precision_list=[];

# 在126个instances中选取126-14=112个。（140-14=126作为训练数据；14作为测试数据）
# 3. ten-fold testing

kf = KFold(n_splits=9, shuffle=True, random_state=1)

for train_index, test_index in kf.split(range(0,126)):
    x_train = x_left[train_index]
    x_train = np.concatenate((x_index0, x_train), axis=0)
    x_test = x_left[test_index]

    y_train = y_left[train_index]
    y_train = np.concatenate((y_index0, y_train), axis=0)
    y_test = y_left[test_index]

    y_train_onehot = y_onehot_left[train_index]
    y_train_onehot = np.concatenate((y_onehot_index0, y_train_onehot), axis=0)
    y_test_onehot = y_onehot_left[test_index]


    # 3. build the triplet loss model
    # base model
    embedding_size = 70
    input_layer = Input((205))
    x = Dense(200, activation="relu")(input_layer)
    x = Dense(150, activation="relu")(x)
    x = Dense(embedding_size, activation="relu")(x) # embedding size=100
    model = Model(input_layer, x)
    # model.summary()
    #
    # triplet loss model
    triplet_model_a = Input((205))
    triplet_model_p = Input((205))
    triplet_model_n = Input((205))
    triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
    triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
    # triplet_model.summary()

    triplet_model.compile(loss=triplet_loss, optimizer="adam")
    triplet_model.fit_generator(data_generator(), steps_per_epoch=50, epochs=50) # 开始训练
    # triplet_model.save("triplet_2.h5")

    # triplet_model = load_model("triplet_2.h5", compile=False)
    # triplet_model.compile(loss=triplet_loss, optimizer="adam") # 对模型进行编译

    # model_embedings = triplet_model.layers[-2].predict(x_data, verbose=1) # 得到相应的 embedding
    # print(model_embedings.shape)

    # reduce the embeddings for visulization
    # reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='correlation').fit_transform(model_embedings)
    # print(reduced_embeddings.shape)

    # fig1 = plt.figure(1) # 搭配plt.close() 使用
    # plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=y_data)
    # plt.show()

    # 4. build the neural network which utilizes the embeddings to classify the self-caring class
    input_layer = Input((embedding_size))
    midlle_value = Dense(100, activation="relu")(input_layer)
    midlle_value = Dense(50, activation="relu")(midlle_value)
    model_output = Dense(7, activation="softmax")(midlle_value) # the model output
    class_model = Model(input_layer, model_output)
    # class_model.summary()

    # the training dataset
    embedings_train = triplet_model.layers[-2].predict(x_train, verbose=1) # 得到相应的 embeddings of training set
    x_train = embedings_train
    y_train =  y_train_onehot # 这里采用onehot label格式
    embedings_test = triplet_model.layers[-2].predict(x_test, verbose=1) # 得到相应的 embeddings of training set
    x_test = embedings_test
    y_test = y_test_onehot

    class_model.compile(loss=focal_loss, optimizer="adam", metrics=['accuracy', recall_score, precision_score, f1])
    history = class_model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2, validation_data=(x_test, y_test))
    # print(history.history.keys()) # see the keywords

    # see the accuracy
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_f1 = history.history['val_f1']
    val_recall = history.history['val_recall_score']
    val_precision = history.history['val_precision_score']

    val_acc_list.append(val_acc[-1])    # record
    val_f1_list.append(val_f1[-1])
    val_recall_list.append(val_recall[-1])
    val_precision_list.append(val_precision[-1])

# 计算十折测试后 模型在测试集上的表现能力
val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" %recall_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" %precision_mean)

with open('final_method_70.pkl', 'wb') as file:
    pickle.dump(loss, file)

    pickle.dump(val_acc, file)
    pickle.dump(val_acc_list, file)

    pickle.dump(val_f1, file)
    pickle.dump(val_f1_list, file)

    pickle.dump(val_recall, file)
    pickle.dump(val_recall_list, file)

    pickle.dump(val_precision, file)
    pickle.dump(val_precision_list, file)

with open('final_method_70.pkl', 'rb') as file:
    loss = pickle.load(file)

    val_acc = pickle.load(file)
    val_acc_list = pickle.load(file)

    val_f1 = pickle.load(file)
    val_f1_list = pickle.load(file)

    val_recall = pickle.load(file)
    val_recall_list = pickle.load(file)

    val_precision = pickle.load(file)
    val_precision_list = pickle.load(file)


val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" %recall_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" %precision_mean)

# the last performance
plt.subplot(1, 5, 1)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 5, 2)
plt.plot(val_acc, label='validation Acc')
plt.title('validation Acc')
plt.legend()

# the ten-fold performance of the val accuracy
plt.subplot(1, 5, 3)
plt.plot(val_f1, label='validation f1')
plt.title('validation f1')
plt.legend()

plt.subplot(1, 5, 4)
plt.plot(val_acc_list, label='ten-fold performance')
plt.title('validation Acc in ten-fold testing')
plt.legend()

plt.subplot(1, 5, 5)
plt.plot(val_f1_list, label='ten-fold performance')
plt.title('validation f1 in ten-fold testing')
plt.legend()

plt.show()