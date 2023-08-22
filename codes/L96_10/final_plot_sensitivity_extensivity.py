from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlowK

colors = [colormap(i*50) for i in range(5)]

mpl.rcParams['lines.markersize']=20
mpl.rcParams['axes.titlesize']=20
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=20
mpl.rcParams['ytick.labelsize']=20
mpl.rcParams['axes.labelsize']=25

dt=0.05   
dt_solver=0.01
model_dims=[10,20,40]
model_name='L96'
#num_clv=model_dim
seed_num=3

spr=1
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96'
#data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/explanation'.format(model_dim)

os.chdir(data_path)
sigmas=np.array([0.5])
line_types=['solid','dashed','dotted']
color_cycler=cycle(colors)
ls_cycler=cycle(line_types)

plt.figure(figsize=(16,10))
for j,model_dim in enumerate(model_dims):
    num_clv=model_dim
    for i,sigma in enumerate(sigmas):
        os.chdir(data_path+'/L96-{}/seed_3'.format(model_dim))
        clv_cosines=np.rad2deg(np.arccos(np.load('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
        quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
        asymmetric_bar=np.zeros((2,num_clv))
        asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
        asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
        plt.errorbar(x=np.arange(1,num_clv+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20,c=next(color_cycler),ls=next(ls_cycler))

plt.xlabel(r'$i^{th}$ BLV index')
plt.xticks(np.arange(1,num_clv+1,4))
plt.ylabel(r'Relative angle w.r.t true BLV$(\theta)$')
plt.tight_layout()
plt.legend(loc='lower center',ncol=len(sigmas))
plt.savefig('blv_angle_sensitivity_extensivity_seed_{}.png'.format(seed_num))



