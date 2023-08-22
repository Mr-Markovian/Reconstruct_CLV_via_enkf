import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlowK

colors = [colormap(i*50) for i in range(5)]

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=25
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize']=25
mpl.rcParams['ytick.labelsize']=25
mpl.rcParams['axes.labelsize']=25

dt=0.05   
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=11

data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_assim/ob3'.format(model_dim)
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L96_{}'.format(model_dim)
os.chdir(data_path+'/PA')


plt.figure(figsize=(12,8))
mus=np.array([0.3,0.7,1.0])
line_types=['dotted','dashed','solid']

subspace_nums=20
chosen_subspaces=np.array([2,5,10,15,20])
markers=['.','.','.']
mss=[15,20,25]

# Legends for the markers
mss1=[15,20,25]

# for i,sub in enumerate(chosen_subspaces):
#     plt.scatter([], [], color=colors[i], alpha=1, s=150,label='n={}'.format(sub))

for i,size in enumerate(mss1):
    plt.plot([], [], c='k', alpha=0.6,lw=2,linestyle=line_types[i],marker='.',markersize=size,label=r'$\mu={}$'.format(mus[i]))

plt.legend(ncol=7,scatterpoints=1, frameon=True, labelspacing=0.2,loc='upper center')

xloc=[0.5,3.5,8,13,18]

for i,mu in enumerate(mus):
    for j,subspace_num in enumerate(chosen_subspaces):
        clv_cosines=np.load('blv_principal_cosine_mu={}_model_dim={}_subspaces_={}.npy'.format(mu,model_dim,subspace_num))
        quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
        asymmetric_bar=np.zeros((2,subspace_num))
        asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
        asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
        #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
        plt.text(xloc[j],55,'n={}'.format(subspace_num),c=colors[j],fontsize=20,rotation=30)
        plt.errorbar(x=np.arange(1,subspace_num+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,ls=line_types[i],label=r'$n={}$'.format(subspace_num),marker=markers[i],ms=mss[i],c=colors[j])

#plt.title(r'Principal angles for n-dimensional subspaces')
plt.xlabel(r'index $(i) \to$',fontsize=25)
plt.ylim(0.0,1.1)
plt.yticks(np.arange(0,100,10))
plt.xticks(np.arange(1,subspace_num+1,2))
#plt.yticks()
plt.ylabel(r' Principal angles $(\theta_i) \to$',fontsize=25)
#plt.legend(loc='upper center',ncol=subspace_nums )
plt.tight_layout()
os.chdir(plots_path)

plt.savefig('blv_principals_3_point_summary_angle_sensitivity_L96-{}_analysis_subspaces_{}.png'.format(model_dim,subspace_num),dpi=300)



