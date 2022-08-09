import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# ls = []
# for i in range(3):
#     ls.append(torch.from_numpy(np.random.rand(2)))
# # print(ls)
# a=torch.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],dtype=float)
# print(nn.functional.normalize(a,p=2.0, dim = 1))
a = np.random.rand(10)
b = np.random.rand(10)
c = a+b
dd = [1,2,3]
print(min(dd))
plt.plot(a)
plt.plot(b)
plt.show()
