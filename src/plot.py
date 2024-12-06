import numpy as np
import matplotlib.pyplot as plt

central = np.load('../save/fed_mnist_cnn_250_iid[0]_M[0]_Central[1]_new.npz')
random = np.load('../save/fed_mnist_cnn_250_iid[0]_M[0]_Central[0]_new.npz')
modular = np.load('../save/fed_mnist_cnn_250_iid[0]_M[1]_Central[0]_new.npz')

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

# Plot Average Accuracy vs Communication rounds
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