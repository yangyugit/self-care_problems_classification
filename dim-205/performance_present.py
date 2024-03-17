import numpy as np
import pickle

with open('care2vec_205_performance.pkl', 'rb') as file:
    loss = pickle.load(file)

    val_acc = pickle.load(file)
    val_acc_list = pickle.load(file)

    val_f1 = pickle.load(file)
    val_f1_list = pickle.load(file)

    val_recall = pickle.load(file)
    val_recall_list = pickle.load(file)

    val_precision = pickle.load(file)
    val_precision_list = pickle.load(file)

print('----------The performance of care2vec ----------')
val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" % precision_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" % recall_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)

with open('neural_network_performance.pkl', 'rb') as file:
    loss = pickle.load(file)

    val_acc = pickle.load(file)
    val_acc_list = pickle.load(file)

    val_f1 = pickle.load(file)
    val_f1_list = pickle.load(file)

    val_recall = pickle.load(file)
    val_recall_list = pickle.load(file)

    val_precision = pickle.load(file)
    val_precision_list = pickle.load(file)

print('----------The performance of mlp ----------')
val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" % precision_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" % recall_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)

with open('neural_network_triplet_205_performance.pkl', 'rb') as file:
    loss = pickle.load(file)

    val_acc = pickle.load(file)
    val_acc_list = pickle.load(file)

    val_f1 = pickle.load(file)
    val_f1_list = pickle.load(file)

    val_recall = pickle.load(file)
    val_recall_list = pickle.load(file)

    val_precision = pickle.load(file)
    val_precision_list = pickle.load(file)

print('----------The performance of triplet ----------')
val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" % precision_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" % recall_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)

with open('final_method_205.pkl', 'rb') as file:
    loss = pickle.load(file)

    val_acc = pickle.load(file)
    val_acc_list = pickle.load(file)

    val_f1 = pickle.load(file)
    val_f1_list = pickle.load(file)

    val_recall = pickle.load(file)
    val_recall_list = pickle.load(file)

    val_precision = pickle.load(file)
    val_precision_list = pickle.load(file)

print('----------The performance of focal ----------')
val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("test acc mean is: %f" % acc_mean)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)
precision_std = np.std(val_performance4)
print("test precision mean is: %f" % precision_mean)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)
recall_std = np.std(val_performance3)
print("test recall mean is: %f" % recall_mean)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("test f1 mean is: %f" % f1_mean)