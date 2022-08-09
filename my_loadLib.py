import scipy.io
import torch
import numpy as np

def loaddatafile(path, L_w):
    data_mat = scipy.io.loadmat(path)
    cnt = np.transpose(np.array(data_mat['cnt'],dtype='f'))
        # transpose row and colomn: now row is each channel time series data.
    list_dataInWindow = []
    # print(len(list_dataInWindow))
    u = 0
    for data_channel in cnt:
        data_window = [0.1*data_channel[i:i + L_w] for i in range(0, len(data_channel), L_w)]
        data_window.pop()
        list_dataInWindow.extend(data_window)
    print(f"there is {len(list_dataInWindow)} windows in the file")
    return list_dataInWindow

def loadTrainingSet(L_w):
    list_dataInWindow1 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1a.mat', L_w)
    list_dataInWindow2 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1b.mat', L_w)
    list_dataInWindow3 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1c.mat', L_w)
    list_dataInWindow4 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1d.mat', L_w)
    list_dataInWindow5 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1e.mat', L_w)
    list_dataInWindow6 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1f.mat', L_w)
    list_dataInWindow7 = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1g.mat', L_w)
    return list_dataInWindow1+list_dataInWindow2+list_dataInWindow3+list_dataInWindow4+list_dataInWindow5+list_dataInWindow6+list_dataInWindow7

def loadTestingSet(L_w):
    list_dataInWindow1 = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1a.mat', L_w)
    list_dataInWindow2 = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1b.mat', L_w)
    list_dataInWindow3 = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1c.mat', L_w)
    list_dataInWindow4 = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1d.mat', L_w)
    list_dataInWindow5 = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1e.mat', L_w)
    return list_dataInWindow1+list_dataInWindow2+list_dataInWindow3+list_dataInWindow4+list_dataInWindow5

