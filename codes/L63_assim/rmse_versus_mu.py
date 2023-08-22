import numpy as np
import os
import matplotlib.pyplot as plt

mus=np.array([0.1,0.3,0.5,0.7,0.9])

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63_assim/ob5'
os.chdir(data_path)

# true trajectory
os.chdir('state')
base_traj=np.load('state.npy')[5000:]
os.chdir('..')

# The rmse array
rmse=np.zeros_like(mus)

for i,mu in enumerate(mus):
    os.chdir(data_path+'/ebias=6.0_ecov=2.0_obs=1_ens=25_mu={}_gap=0.01_alpha=1.0_loc=none_r=0'.format(mu))
    analysis_traj=np.load('filtered_mean.npy')[5000:]
    rmse[i]=np.sqrt(np.mean(np.sum((analysis_traj-base_traj)**2,axis=1),axis=0))

plt.figure()
plt.scatter(mus,rmse)
plt.show()
os.chdir(data_path)
np.save('rmse_versus_mu.npy',rmse)
