import pickle
import numpy as np

# 1. the neural network performance
with open('neural_network_performance.pkl', 'rb') as file:
  loss = pickle.load(file)
  val_acc = pickle.load(file)
  val_f1 = pickle.load(file)

  val_acc_list = pickle.load(file)
  val_f1_list = pickle.load(file)

val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("Neural network performance -> test acc mean is: %f" %acc_mean)
print("Neural network performance -> test acc std is: %f:" %acc_std)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("Neural network performance -> test f1 mean is: %f" %f1_mean)
print("Neural network performance -> test f2 std is: %f" %f1_std)

print('===========================================================')

# 2. the neural network with focal loss performance
with open('neural_network_focal_performance.pkl_performance.pkl', 'rb') as file:
  loss = pickle.load(file)
  val_acc = pickle.load(file)
  val_f1 = pickle.load(file)

  val_acc_list = pickle.load(file)
  val_f1_list = pickle.load(file)

val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("Neural network focal performance -> test acc mean is: %f" %acc_mean)
print("Neural network focal performance -> test acc std is: %f:" %acc_std)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("Neural network focal performance -> test f1 mean is: %f" %f1_mean)
print("Neural network focal performance -> test f1 std is: %f" %f1_std)


print('===========================================================')

# 3. the neural network with triplet loss performance
with open('neural_network_triplet_performance.pkl', 'rb') as file:
  loss = pickle.load(file)
  val_acc = pickle.load(file)
  val_f1 = pickle.load(file)

  val_acc_list = pickle.load(file)
  val_f1_list = pickle.load(file)

val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("Neural network triplet performance -> test acc mean is: %f" %acc_mean)
print("Neural network triplet performance -> test acc std is: %f:" %acc_std)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("Neural network triplet performance -> test f1 mean is: %f" %f1_mean)
print("Neural network triplet performance -> test f1 std is: %f" %f1_std)


print('===========================================================')

# 4. the neural network with triplet loss and focal loss performance
with open('final_method.pkl', 'rb') as file:
  loss = pickle.load(file)
  val_acc = pickle.load(file)
  val_f1 = pickle.load(file)

  val_acc_list = pickle.load(file)
  val_f1_list = pickle.load(file)

val_performance1 = np.array(val_acc_list)
acc_mean = np.mean(val_performance1)
acc_std = np.std(val_performance1)
print("final_method performance -> test acc mean is: %f" %acc_mean)
print("final_method performance -> test acc std is: %f:" %acc_std)

val_performance2 = np.array(val_f1_list)
f1_mean = np.mean(val_performance2)
f1_std = np.std(val_performance2)
print("final_method performance -> test f1 mean is: %f" %f1_mean)
print("final_method performance -> test f1 std is: %f" %f1_std)
