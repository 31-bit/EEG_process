import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from my_functionLib import plot_kurtosis, plot_EEG_windows, plot_Compared_EEG
import torch
from scipy.stats import kurtosis
from my_loadLib import loaddatafile, loadTrainingSet
from model import NeuralNetwork
import torch.nn as nn

n = 100 # its length of the training sample
L_w = 2*n # length of windows = integral multiple of n
# list_dataInWindow = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1a.mat', L_w)  # todo need load more sample for training
list_dataInWindow = loadTrainingSet(L_w)
list_kur = []
for i in list_dataInWindow:
    list_kur.append(kurtosis(i))

numbersOfWindow = 20
i_window = 10
plt.figure("kurtosisPlot")
plot_kurtosis(list_kur, numbersOfWindow, i_window)
plt.figure("EEG_plot_data", figsize=(16, 4))
plot_EEG_windows(list_dataInWindow, numbersOfWindow, i_window)

# delete the EEG contains OAs
th_kur = 3.5
dataset_NOE = []
X = []
for i in range(len(list_kur)):
    if list_kur[i] < th_kur:
        dataset_NOE.append(list_dataInWindow[i])  # non-normalised, for plot comparision graph
        X.append(torch.from_numpy(list_dataInWindow[i][0:100]))
        X.append(torch.from_numpy(list_dataInWindow[i][100:200]))
        # print(type(dataset_NOE[0]))
plt.figure("EEG_NOE", figsize=(16, 4))
plot_EEG_windows(dataset_NOE, numbersOfWindow, i_window)


# construct the model
device = "mps" if torch.has_mps else "cpu"
print(f"Using {device} device")
X = nn.functional.normalize(torch.stack(X).to(device))
Y = X
# print(len(dataset_NOE))
# print(type(dataset_NOE[0]))
model = NeuralNetwork().to(device)
loss = nn.MSELoss()
optimizater = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 10
ls_loss = []
for i in range(epochs):
    Y_est = model(X)
    l = loss(Y, Y_est)
    ls_loss.append(l.to('cpu').detach().numpy())
    l.backward()
    optimizater.step()
    optimizater.zero_grad()
    print(f"epochs: {i}, cost is {l}")
for para in model.parameters():  # todo need to print the parameter with details name
    print(para.name, para.data)

plt.figure("comparision of containminated and reconstructed EEG", figsize=(16, 4))
plot_Compared_EEG(X.to('cpu').detach().numpy(), Y_est.to('cpu').detach().numpy(), numbersOfWindow, i_window)
ls_epoch = [i for i in range(epochs)]
plt.figure("cost function over epochs")
plt.plot(ls_epoch, ls_loss)
plt.show()

### Testing
# todo: add step to save and reload the model.
list_TestingDataInWindow = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1b.mat', L_w)
# todo need load more sample for testing
# select a NOE as reference
list_ctn = []
list_noe = []
index_NOE = []
index_ctd = []
list_kur_test = []
for i in range(len(list_TestingDataInWindow)):
    j = list_TestingDataInWindow[i]
    kur = kurtosis(j)
    list_kur_test.append(kurtosis(kur))
    if kur > th_kur:
        index_ctd.append(2*i)
        index_ctd.append(2*i+1)
        list_ctn.append(j[0:100])
        list_ctn.append(j[100:200])
    else:
        index_NOE.append(2*i)
        index_NOE.append(2*i+1)
        list_noe.append(torch.from_numpy(j[0:100]))
        list_noe.append(torch.from_numpy(j[100:200]))
# torch.from_numpy(j)
# standarlizaiton
# initialise the reference
reference = list_TestingDataInWindow[index_NOE[3]]
SN_min = min(reference)
SN_max = max(reference)
X_std = []
list_min = []
for i in list_ctn:
    temp_list = []
    for ctn in i:
        temp_list.append((ctn-SN_min)/(SN_max-SN_min))
    min_temp = min(temp_list)
    list_min.append(min_temp)
    new_window = [k-min_temp for k in temp_list]
    X_std.append(torch.tensor(new_window))

print(1)
X_std = nn.functional.normalize(torch.stack(X_std).to(device))
Y_opt = model(X_std)


X_noe = nn.functional.normalize(torch.stack(list_noe).to(device))
Y_noe = model(X_std)
# reconstruct
print(len(X_std))
EEG_opt = Y_opt.to('cpu').detach().numpy()
print(len(Y_opt))
print(len(X_std[0]))
list_crt = []

for i in range(len(EEG_opt)):
    temp_min = list_min[i]
    crt = []
    for opt_k in EEG_opt[i]:
        crt_k = (opt_k + temp_min)*(SN_max -SN_min)+SN_min
        crt.append(crt_k)
    list_crt.append(crt)

loss(s, Y_est)
loss(Y, Y_est)