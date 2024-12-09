import numpy as np
import matplotlib.pyplot as plt
import sys

central = np.load('../save/fed_mnist_cnn_250_iid[0]_M[0]_Central[1]_new.npz')
random = np.load('../save/fed_mnist_cnn_250_iid[0]_M[0]_Central[0]_new.npz')
modular = np.load('../save/fed_mnist_cnn_250_iid[0]_M[1]_Central[0]_new.npz')

# plot time per communication round
t_c = central['cumulative_time']
t_r = random['cumulative_time']
t_m = modular['cumulative_time']
temp_c, temp_r, temp_m = np.zeros(len(t_c)), np.zeros(len(t_c)), np.zeros(len(t_c))
temp_c[0], temp_r[0], temp_m[0] = t_c[0], t_r[0], t_m[0]
for i in range(1, len(t_c)):
    temp_c[i], temp_r[i], temp_m[i] = t_c[i] - t_c[i-1], t_r[i] - t_r[i-1], t_m[i] - t_m[i-1]
plt.figure()
plt.plot(temp_c, linewidth=3)
plt.plot(temp_r, linestyle='dashed')
plt.plot(temp_m, linestyle='dotted')
plt.ylabel('Communication Time (sec)')
plt.xlabel('Communication Rounds')
plt.legend(['Centralized', 'Random', 'Modularity'])
plt.title('Communication Time Per Round')
plt.grid()
plt.savefig('../save/time.png')

# plot cumulative battery used
plt.figure()
plt.title('Avg. Battery Use vs Communication Round - MNIST')
plt.plot(central['battery'])
plt.plot(random['battery'])
plt.plot(modular['battery'])
plt.ylabel('Avg. Battery Percentage Used')
plt.xlabel('Communication Rounds')
plt.legend(['Centralized', 'Random', 'Modularity'])
plt.grid()
plt.savefig('../save/battery_loss_mnist_250.png')

# plot training loss
plt.figure()
plt.title('Training Loss vs Communication Rounds - MNIST')
plt.plot(central['train_loss'])
plt.plot(random['train_loss'])
plt.plot(modular['train_loss'])
plt.ylabel('Training Loss')
plt.xlabel('Communication Rounds')
plt.legend(['Centralized', 'Random', 'Modularity'])
plt.grid()
plt.savefig('../save/train_loss_mnist_250.png')