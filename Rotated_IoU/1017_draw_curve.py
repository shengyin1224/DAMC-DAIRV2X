import numpy as np
import matplotlib.pyplot as plt

fontsize=20
fig, ax = plt.subplots()
fig.set_size_inches(7.5, 5.7)

# fig = plt.figure()
# fig.set_size_inches(7.5, 5.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Background color
ax.set_facecolor('#eaeaf1')
plt.grid(color='white', linestyle='-', linewidth=1)
ax.set_axisbelow(True)

fedavg = np.array([68.47, 66.00, 63.63, 57.86, 50.14])
fedavg_disco = np.array([69.95, 67.55, 64.44, 60.45, 53.85])

feddyn = np.array([70.14, 69.44, 66.58, 63.24, 55.60])
feddyn_disco = np.array([72.05, 70.48, 67.12, 64.33, 58.12])

X_axis = np.arange(len(fedavg))

plt.plot(X_axis, fedavg, color='burlywood', label = 'FedAvg', marker='s', linestyle='dashed')
plt.plot(X_axis, fedavg_disco, color='burlywood', label = 'FedAvg+Disco', marker='s')
plt.plot(X_axis, feddyn, color='slateblue', label = 'FedDyn', marker='s', linestyle='dashed')
plt.plot(X_axis, feddyn_disco, color='slateblue', label = 'FedDyn+Disco', marker='s')


# plt.bar(X_axis - 0.3, fedavg_disco-fedavg, 0.2, color='#005f73', label = 'FedAvg')
# plt.bar(X_axis - 0.1, fedprox_disco-fedprox, 0.2, color='#0a9396', label = 'FedProx')
# plt.bar(X_axis + 0.1, feddyn_disco-feddyn, 0.2, color='#83c5be', label = 'FedDyn')
# plt.bar(X_axis + 0.3, moon_disco-moon, 0.2, color='#d1b3c4', label = 'MOON')

# plt.legend(fontsize=15, loc='center left')
plt.xticks(X_axis, [1, 2, 5, 10, 20], fontsize=15, fontweight='medium')
plt.yticks(fontsize=15, fontweight='medium')
plt.xlabel('Globally Imbalance Level', fontsize=fontsize, fontweight='medium')
plt.title('Accuracy (%)', fontsize=fontsize, fontweight='medium')

# plt.plot(fedavg_disco-fedavg, label='FedAvg', color='r', marker='s')
# plt.plot(fedprox_disco-fedprox, label='FedProx', color='b', marker='s')
# plt.plot(feddyn_disco-feddyn, label='FedDyn', color='k', marker='s')
# plt.plot(moon_disco-moon, label='MOON', color='deeppink', marker='s')



# plt.plot(fedavg, label='FedAvg', color='r')
# plt.plot(fedavg_disco, label='FedAvg+Disco', color='r', linestyle='dashed')

# plt.plot(fedprox, label='FedProx', color='b')
# plt.plot(fedprox_disco, label='FedProx+Disco', color='b', linestyle='dashed')

# plt.plot(feddyn, label='FedDyn', color='deeppink')
# plt.plot(feddyn_disco, label='FedDyn+Disco', color='deeppink', linestyle='dashed')

# plt.plot(moon, label='MOON', color='k')
# plt.plot(moon_disco, label='MOON+Disco', color='k', linestyle='dashed')

plt.savefig('Rotated_IoU/ablation_globally_imbalanced.png', dpi=300)