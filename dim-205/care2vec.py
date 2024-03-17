# 1. load the self-care dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle

import random # the random function
import matplotlib.pyplot as plt # plot like matlab
# import umap # dimensionality reduction package
from tensorflow.keras import backend as K  #custom loss function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score,f1_score

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
print(dict)

x_data = x
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
x_index0 = x_data[index_0]; # y_index0 = y_data[index_0];
y_onehot_index0 = y_onehot[index_0];
x_left = np.delete(x_data, index_0, axis=0); # y_left = np.delete(y_data, index_0, axis=0);
y_onehot_left = np.delete(y_onehot, index_0, axis=0)


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

kf = KFold(n_splits=10, shuffle=True, random_state=1)

for train_index, test_index in kf.split(range(0,70)):
    x_train = x_left[train_index]
    x_train = np.concatenate((x_index0, x_train), axis=0)
    x_test = x_left[test_index]

    y_train = y_onehot_left[train_index]
    y_train = np.concatenate((y_onehot_index0, y_train), axis=0)
    y_test = y_onehot_left[test_index]

    # 3. build classfication model
    input_layer = Input((205))
    midlle_value = Dense(205, activation="relu")(input_layer)
    model_output = Dense(7, activation="softmax")(midlle_value)  # the model output
    class_model = Model(input_layer, model_output)
    # class_model.summary()

    class_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', recall_score, precision_score, f1])
    history = class_model.fit(x_train, y_train, epochs=50, batch_size=140, verbose=2, validation_data=(x_test, y_test))
    print(history.history.keys()) # see the keywords

    # see the accuracy (of train and test data both)
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

with open('care2vec_205_performance.pkl', 'wb') as file:
    pickle.dump(loss, file)

    pickle.dump(val_acc, file)
    pickle.dump(val_acc_list, file)

    pickle.dump(val_f1, file)
    pickle.dump(val_f1_list, file)

    pickle.dump(val_recall, file)
    pickle.dump(val_recall_list, file)

    pickle.dump(val_precision, file)
    pickle.dump(val_precision_list, file)



# # the last performance
# plt.subplot(1, 5, 1)
# plt.plot(loss, label='Training Loss')
# plt.title('Training Loss')
# plt.legend()
#
# plt.subplot(1, 5, 2)
# plt.plot(val_acc, label='validation Acc')
# plt.title('validation Acc')
# plt.legend()
#
# # the ten-fold performance of the val accuracy
# plt.subplot(1, 5, 3)
# plt.plot(val_f1, label='validation f1')
# plt.title('validation f1')
# plt.legend()
#
# plt.subplot(1, 5, 4)
# plt.plot(val_acc_list, label='ten-fold performance')
# plt.title('validation Acc in ten-fold testing')
# plt.legend()
#
# plt.subplot(1, 5, 5)
# plt.plot(val_f1_list, label='ten-fold performance')
# plt.title('validation f1 in ten-fold testing')
# plt.legend()
#
# plt.show()