import numpy as np
import pickle
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.ticker as mtick

index = ['20', '30', '34', '40', '50', '56', '60', '70', '80', '90', '100', '125', '150', '175', '205']

# the triplet metrics
triplet_acc = []
triplet_pre = []
triplet_recall = []
triplet_f1 = []
for i in index:
    with open('./dim-' + i + '/' + 'neural_network_triplet_' + i + '_performance.pkl', 'rb') as file:
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

    val_performance2 = np.array(val_f1_list)
    f1_mean = np.mean(val_performance2)

    val_performance3 = np.array(val_recall_list)
    recall_mean = np.mean(val_performance3)

    val_performance4 = np.array(val_precision_list)
    precision_mean = np.mean(val_performance4)

    triplet_acc.append(acc_mean)
    triplet_pre.append(precision_mean)
    triplet_recall.append(recall_mean)
    triplet_f1.append(f1_mean)

# the focal metrics
focal_acc = []
focal_pre = []
focal_recall = []
focal_f1 = []
for i in index:
    with open('./dim-' + i + '/' + 'final_method_' + i + '.pkl', 'rb') as file:
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

    val_performance2 = np.array(val_f1_list)
    f1_mean = np.mean(val_performance2)

    val_performance3 = np.array(val_recall_list)
    recall_mean = np.mean(val_performance3)

    val_performance4 = np.array(val_precision_list)
    precision_mean = np.mean(val_performance4)

    focal_acc.append(acc_mean)
    focal_pre.append(precision_mean)
    focal_recall.append(recall_mean)
    focal_f1.append(f1_mean)

# mlp metrics
mlp_acc = []
mlp_pre = []
mlp_recall = []
mlp_f1 = []

with open('./dim-205' + '/' + 'neural_network_performance.pkl', 'rb') as file:
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

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)

val_performance3 = np.array(val_recall_list)
recall_mean = np.mean(val_performance3)

val_performance4 = np.array(val_precision_list)
precision_mean = np.mean(val_performance4)

mlp_acc.append(acc_mean); mlp_acc = np.ones(np.size(index)) * mlp_acc
mlp_pre.append(precision_mean); mlp_pre = np.ones(np.size(index)) * mlp_pre
mlp_recall.append(recall_mean); mlp_recall = np.ones(np.size(index)) * mlp_recall
mlp_f1.append(f1_mean); mlp_f1 = np.ones(np.size(index)) * mlp_f1

plt.rc('font',family='Times New Roman')

plt.figure(num=1, dpi=600)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.plot(index, np.array(mlp_acc) * 100,  marker='o', alpha=0.8 ,linestyle='-', linewidth=2.5, color='g', markeredgecolor='darkgreen',markersize='5',markeredgewidth=7)
plt.plot(index, np.array(triplet_acc) * 100,  marker='o',  alpha=0.8 , linestyle='-', linewidth=2.5, color='b', markeredgecolor='midnightblue',markersize='5',markeredgewidth=7)
plt.plot(index, np.array(focal_acc) * 100,  marker='o', alpha=0.8 , linestyle='-', linewidth=2.5, color='r', markeredgecolor='firebrick',markersize='5',markeredgewidth=7)
plt.legend(loc='best', labels=['MLP', 'Triplet', 'Focal'], fontsize=18)
plt.xlabel('Dimensions of feature extraction', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
plt.xticks(rotation=70)
plt.tick_params(labelsize=18)
plt.savefig('acc', bbox_inches='tight')
plt.clf()
print('-------accuracy---------')
print('mlp acc: %f' % np.mean(mlp_acc))
print('triplet acc: %f' % np.mean(triplet_acc))
print('focal acc: %f' % np.mean(focal_acc))

plt.figure(num=2, dpi=600)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.plot(index, mlp_pre, marker='o', alpha=0.8 ,linestyle='-', linewidth=2.5, color='g', markeredgecolor='darkgreen',markersize='5',markeredgewidth=7)
plt.plot(index, triplet_pre, marker='o',  alpha=0.8 , linestyle='-', linewidth=2.5, color='b', markeredgecolor='midnightblue',markersize='5',markeredgewidth=7)
plt.plot(index, focal_pre, marker='o', alpha=0.8 , linestyle='-', linewidth=2.5, color='r', markeredgecolor='firebrick',markersize='5',markeredgewidth=7)
# plt.legend(loc='upper right', labels=['MLP', 'triplet', 'focal'])
plt.xlabel('Dimensions of feature extraction', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(rotation=70)
plt.tick_params(labelsize=18)
plt.savefig('precision', bbox_inches='tight')
plt.clf()
print('-------precision---------')
print('mlp acc: %f' % np.mean(mlp_pre))
print('triplet acc: %f' % np.mean(triplet_pre))
print('focal acc: %f' % np.mean(focal_pre))

plt.figure(num=3, dpi=600)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.plot(index, mlp_recall, marker='o', alpha=0.8 ,linestyle='-', linewidth=2.5, color='g', markeredgecolor='darkgreen',markersize='5',markeredgewidth=7)
plt.plot(index, triplet_recall, marker='o',  alpha=0.8 , linestyle='-', linewidth=2.5, color='b', markeredgecolor='midnightblue',markersize='5',markeredgewidth=7)
plt.plot(index, focal_recall, marker='o', alpha=0.8 , linestyle='-', linewidth=2.5, color='r', markeredgecolor='firebrick',markersize='5',markeredgewidth=7)
# plt.legend(loc='upper right', labels=['MLP', 'triplet', 'focal'])
plt.xlabel('Dimensions of feature extraction', fontsize=18)
plt.ylabel('Recall', fontsize=18)
plt.xticks(rotation=70)
plt.tick_params(labelsize=18)
plt.savefig('Recall', bbox_inches='tight')
plt.clf()
print('-------recall---------')
print('mlp acc: %f' % np.mean(mlp_recall))
print('triplet acc: %f' % np.mean(triplet_recall))
print('focal acc: %f' % np.mean(focal_recall))

plt.figure(num=4, dpi=600)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.plot(index, mlp_f1, marker='o', alpha=0.8 ,linestyle='-', linewidth=2.5, color='g', markeredgecolor='darkgreen',markersize='5',markeredgewidth=7)
plt.plot(index, triplet_f1, marker='o',  alpha=0.8 , linestyle='-', linewidth=2.5, color='b', markeredgecolor='midnightblue',markersize='5',markeredgewidth=7)
plt.plot(index, focal_f1,marker='o', alpha=0.8 , linestyle='-', linewidth=2.5, color='r', markeredgecolor='firebrick',markersize='5',markeredgewidth=7)
# plt.legend(loc='upper right', labels=['MLP', 'triplet', 'focal'])
plt.xlabel('Dimensions of feature extraction', fontsize=18)
plt.ylabel('F1 score', fontsize=18)
plt.xticks(rotation=70)
plt.tick_params(labelsize=18)
plt.savefig('F1 score', bbox_inches='tight')
plt.clf()
print('-------F1 score---------')
print('mlp acc: %f' % np.mean(mlp_f1))
print('triplet acc: %f' % np.mean(triplet_f1))
print('focal acc: %f' % np.mean(focal_f1))
