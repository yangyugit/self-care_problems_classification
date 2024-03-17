import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# load the self-caring data
data_file = './SCADI-Dataset/SCADI.csv'
original_data = pd.read_csv(data_file)

# the age of the dataset
age = original_data.loc[:,'Age'].tolist()   # list

age_dict = {}   # dict
age_set = set(age)  # 去除list中的重复项
for item in age_set:
    age_dict.update({item: age.count(item)})
values = []; label = [];
# values = list(age_dict.values())
# label = list(age_dict.keys())
for i,j in age_dict.items():
    label.append(i)
    values.append(j)

print('-----samples of different age-----')
print(values)
print('-----the corresponding age-----')
print(label)
print('------total samples-----')
print(sum(values)) # total samples

plt.rc('font',family='Times New Roman')
plt.figure(num=1,dpi=200)
explode=np.ones(len(values)) * 0.01
print(values)
print(label)
plt.pie(values,colors=cm.ScalarMappable().to_rgba(values), explode=explode,labels=label,autopct='%1.1f%%')  #绘制饼图
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(values), vmax=max(values)))
cb = plt.colorbar(sm)
cb.ax.tick_params(labelsize=10)
cb.set_label(label='The samples of different ages',loc='top')   #loc参数
plt.title('The proportions of different ages in the SCADI dataset', fontdict={'family' : 'Times New Roman'})
plt.show()
plt.savefig('SCADIa', bbox_inches = 'tight')
plt.clf()

# 计算分位数和四分位数
ages = label
q1_a = np.percentile(ages, 25)
q3_a = np.percentile(ages, 75)
q_a = np.median(ages)
print('全样本年龄统计信息')
print(q_a)
print(q1_a)
print(q3_a)
print(q3_a - q1_a)



# the class of the dataset
class1 = original_data.loc[:,'Classes'].tolist()

class1_dict = {}
class1_set = set(class1)
for item in class1_set:
    class1_dict.update({item: class1.count(item)})

label = []; values = [];
for i,j in class1_dict.items():
    label.append(i)
label.sort()
for i in label:
    values.append(class1_dict[i])
print('--------class label--------')
print(label)
print('--------samples of different class-------')
print(values)

plt.figure(num=2, dpi=600)

for i in range(len(label)):
    plt.bar(label[i],values[i],color=(0.1,0.3,0.1*i))

plt.title('The samples of different categories in the SCADI dataset', fontdict={'family' : 'Times New Roman', 'size': 18})
plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.xlabel('Classes', fontproperties = 'Times New Roman', size = 18)
plt.ylabel('Number of samples', fontproperties = 'Times New Roman', size = 18)
plt.savefig('SCADIb', bbox_inches = 'tight')
plt.clf()


