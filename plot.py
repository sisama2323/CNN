import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt

test_1_file = 'CNN/netmine_accuracy_test1.txt'
test_5_file = 'CNN/netmine_accuracy_test5.txt'
val_1_file = 'CNN/netmine_accuracy_val1.txt'
val_5_file = 'CNN/netmine_accuracy_val5.txt'


test_1 = []
with open(test_1_file, 'r') as f:
    for i, line in enumerate(f):
        test_1.append(float(line)) 


test_5 = []
with open(test_5_file, 'r') as f:
    for i, line in enumerate(f):
        test_5.append(float(line)) 

val_1 = []
with open(val_1_file, 'r') as f:
    for i, line in enumerate(f):
        val_1.append(float(line)) 

val_5 = []
with open(val_5_file, 'r') as f:
    for i, line in enumerate(f):
        val_5.append(float(line)) 

plt.figure(1)
plt.plot(test_1, label='Test set Top1')
plt.plot(test_5, label='Test set Top5')
plt.plot(val_1, '--', label='Validation set Top1')
plt.plot(val_5, '--', label='Validation set Top5')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Net1')
plt.show()


a=5





