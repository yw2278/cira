
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use("ggplot")
get_ipython().magic('matplotlib inline')

path = "Output/"
frames = np.load(path + "IR_temp.npy")
fr_t = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
fr_t = fr_t.T
fr_t.shape
tem_fr = 63 - fr_t
temp_pca = PCA()
temp_pca.fit(tem_fr)
plt.figure(1, figsize=(15, 9))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(temp_pca.explained_variance_ratio_[:10], linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

mu = np.mean(tem_fr, axis=0)
X_tran = temp_pca.transform(tem_fr)
temp_no_sun = np.dot(X_tran[:,1:], temp_pca.components_[1:,])
temp_no_sun += mu
diff = tem_fr - temp_no_sun
np.save("Output/temperature_diff.npy", diff)

