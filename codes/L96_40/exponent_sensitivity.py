"""We plot the mean of cosines of the angles after the BLVs converge.
We also then plot the mean versus clv index for different values of sigma.
We also obtain the exponents and their values"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from cmcrameri import cm
colormap=cm.batlowK

colors = [colormap(i*50) for i in range(5)]

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=25
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize']=25
mpl.rcParams['ytick.labelsize']=25
mpl.rcParams['axes.labelsize']=25

model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=11


data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_seed_{}/Sensitivity'.format(model_dim,seed_num)
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L96_{}'.format(model_dim)

os.chdir(data_path)

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('sigma=0.0')
exp=np.load('local_growth_rates_{}_model_L96_state_noisy.npy'.format(model_dim))
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.

fig,ax=plt.subplots(figsize=(12,8))
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
for i,sigma in enumerate(sigmas):
    os.chdir(data_path+'/sigma={}'.format(sigma))
    exp1=np.load('local_growth_rates_{}_model_L96_state_noisy.npy'.format(model_dim))
    ax.scatter(np.arange(1,num_clv+1),np.mean(exp1,axis=0),label='$\sigma={}$'.format(sigma),marker='*',s=180,color=colors[i])

ax.scatter(np.arange(1,num_clv+1),np.mean(exp,axis=0),label='$truth$',s=100)    
ax.set_xlabel(r'i $ \to$',fontsize=25)
ax.set_xticks(np.arange(1,num_clv+1,3))
ax.set_ylabel(r'$\hat \lambda_i$',fontsize=25)
plt.legend(ncol=3,loc='upper right',fontsize=20)


ax_inset = ax.inset_axes([0.08, 0.08, 0.7, 0.6])#)
#ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
for i,sigma in enumerate(sigmas):
    os.chdir(data_path+'/sigma={}'.format(sigma))
    exp1=np.load('local_growth_rates_{}_model_L96_state_noisy.npy'.format(model_dim))
    ax_inset.scatter(np.arange(1,num_clv+1),np.absolute(np.mean(exp1,axis=0)-np.mean(exp,axis=0)),marker='*',s=160,color=colors[i])
    #plt.scatter(np.arange(1,num_clv+1),np.mean(exp1,axis=0),label='$\sigma={}$'.format(sigma),marker='*',s=190,color=colors[i])
    ax_inset.plot(np.arange(1,num_clv+1),np.absolute(np.mean(exp1,axis=0)-np.mean(exp,axis=0)),color=colors[i])
    # save these 

#ax_inset.set_xlabel(r'i $ \to$',fontsize=15)
ax_inset.set_ylabel(r'$\hat \lambda_i- \lambda_i $',fontsize=25)
#ax_inset.yaxis.set_label_position("right")
ax_inset.yaxis.tick_right()
ax_inset.patch.set_alpha(0.5)
os.chdir(plots_path)
plt.tight_layout()
plt.savefig('L96_{}_exp_sensitivity_seed_{}.pdf'.format(model_dim,seed_num))