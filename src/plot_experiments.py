import numpy as np
import os
import matplotlib.pyplot as plt
import sys

folder = '../save/'

cent_prl = []
mod_prl = []
rand_prl = []
cent_nrr = []
mod_nrr = []
rand_nrr = []

# collect all files saved
for file_name in os.listdir(folder):
    if file_name.endswith(".npz"):  # Check if the file is an .npz file
        file_path = os.path.join(folder, file_name)

        if "_nrr[0.4]" in file_path:
            if "M[0]_Central[0]" in file_path:
                rand_prl.append(file_path)
            elif "M[1]" in file_path:
                mod_prl.append(file_path)
            elif "Central[1]" in file_path:
                cent_prl.append(file_path)
        
        if "_prl[0.4]" in file_path:
            if "M[0]_Central[0]" in file_path:
                rand_nrr.append(file_path)
            elif "M[1]" in file_path:
                mod_nrr.append(file_path)
            elif "Central[1]" in file_path:
                cent_nrr.append(file_path)

# sort files for each collection
cent_prl.sort(), mod_prl.sort(), rand_prl.sort(), cent_nrr.sort(), mod_nrr.sort(), rand_nrr.sort()

lists = [cent_prl, mod_prl, rand_prl, cent_nrr, mod_nrr, rand_nrr]

results = []
for list in lists:
    test_acc = []
    for sim in list:
        data = np.load(sim)
        test_acc.append(float(data['test_acc']))
    results.append(test_acc)

x_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ind = np.arange(len(x_axis))  
width = 0.25

# round failure sweep
plt.figure()
bar1 = plt.bar(ind, results[0], width)
bar2 = plt.bar(ind+width, results[2], width)
bar3 = plt.bar(ind+width*2, results[1], width)

plt.xlabel("Probability of Round Failure")
plt.ylabel("Test Accuracy")
plt.title("Test Performance vs Round Failure")
plt.xticks(ind+width, x_axis)
plt.legend((bar1, bar2, bar3), ('Centralized', 'Random', 'Modularity'), loc = 'lower left')
plt.grid()
plt.savefig("../save/prl_sweep.png")

# node removal sweep
plt.figure()
bar1 = plt.bar(ind, results[3], width)
bar2 = plt.bar(ind+width, results[5], width)
bar3 = plt.bar(ind+width*2, results[4], width)

plt.xlabel("Percentage of Nodes Removed Per Round")
plt.ylabel("Test Accuracy")
plt.title("Test Performance vs Nodes Removed")
plt.xticks(ind+width, np.array(x_axis)*100)
plt.legend((bar1, bar2, bar3), ('Centralized', 'Random', 'Modularity'), loc = 'lower left')
plt.grid()
plt.savefig("../save/nrr_sweep.png")