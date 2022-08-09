import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
def calculate_kurtosis2 (element):
    mean = np.sum(element)/len(element)
    sum = 0
    for i in element:
        sum = sum + i -mean
    kur = pow(sum, 4)/(np.power(np.std(element), 4)*len(element))

    return kur

def calculate_kurtosis (element):
    mean = np.mean(element)
    sum = 0
    for i in range(len(element)):
        element[i] = element[i] - mean

    cm2 = np.mean(np.pow(sum, 2))
    cm4 = np.mean(np.pow(sum, 4))

    kur = cm4 - 3*pow(cm2, 2)

    return kur

def plot_kurtosis(list_kur,numbersOfWindow, i_window):
    index_list = [i for i in range(numbersOfWindow)]
    plt.stem(index_list, list_kur[i_window:i_window+numbersOfWindow])
    plt.title('the kurtosis plot for each window')
    plt.xlabel('window number')
    plt.ylabel('kurtosis value')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数


def plot_EEG_windows(list_dataInWindow, numbersOfWindow, i_window):
    windows10 = list_dataInWindow[i_window:i_window + numbersOfWindow]
    EEG_plot_data = []
    for i in windows10:
        el = np.ndarray.tolist(i)
        EEG_plot_data.extend(el)
    plt.plot(EEG_plot_data, 'r--', label='type1')
    plt.title('A typical contaminated EEG signal for 20s.')
    plt.xlabel('seconds')
    plt.ylabel('')
    x_major_locator = MultipleLocator(len(list_dataInWindow[0]))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数

def plot_Compared_EEG(contaminated_EEG, reconstruct_EEG, numbersOfWindow, i_window):
    windows1 = contaminated_EEG[i_window:i_window + numbersOfWindow]
    windows2 = reconstruct_EEG[i_window:i_window + numbersOfWindow]
    plot_con = []
    plot_re = []
    for i in range(len(windows1)):
        plot_con.extend(np.ndarray.tolist(windows1[i]))
        plot_re.extend(np.ndarray.tolist(windows2[i]))
    plt.plot(plot_con, 'r--', label='contaminated EEG')
    plt.plot(plot_re, 'g--', label='Reconstructed EEG')
    plt.title('A comparision of EEG signal for 20s.')
    plt.xlabel('seconds')
    plt.ylabel('')
    x_major_locator = MultipleLocator(len(contaminated_EEG[0]))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
