import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
path = "data/IRTL_frames/"
fnames = glob.glob(path + "*.npy")
frames = np.array([np.load(i) for i in fnames])
temp_frame = frames[0][28:696, 990, :]
temprature = np.linspace(16,47, len(temp_frame))
lr = LinearRegression(n_jobs=-1)
lr.fit(temp_frame, temprature)
## 968
def cal_temp(data):
    temp = data[:, :968, :]
    temp_2 = temp.reshape(temp.shape[0]*temp.shape[1], temp.shape[2])
    temp_ = lr.predict(temp_2)
    return temp_.reshape(temp.shape[0], temp.shape[1])

out_temp = np.array(list(map(cal_temp, frames)))
print(out_temp.shape)