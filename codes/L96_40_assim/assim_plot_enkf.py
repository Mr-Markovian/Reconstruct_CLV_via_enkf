import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns
from cmcrameri import cm

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

# Change here for different observation gap and observation covariance parameter
mu=0.3
ob_gap=0.05
model_dim=40
k=3
N=25 
loc_fun='gaspri'
l_scale=4
alpha=1.0
ob_dim=20
ecov=1.0 

ebias=-5.0     

data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_assim'.format(model_dim)
os.chdir(data_path)

#load the state
state=np.load('Multiple_trajectories_N=1_gap=0.05_ti=0.0_tf=1050.0_dt_0.05_dt_solver=0.01.npy')


os.chdir(data_path+'/ob{}'.format(k)) 
obs=np.load('ob{}_gap_{}_H2__mu={}_obs_cov1.npy'.format(k,ob_gap,mu))

#Go inside the data folder......................................
folder_label='ebias={}_ecov={}_obs={}_ens={}_mu={}_gap={}_alpha={}_loc=gaspri_r={}'.format(ebias, ecov,ob_dim,N,mu,ob_gap,alpha,l_scale)
print(os.getcwd())
os.chdir(folder_label)

#Load data....
a_ens=np.load('filtered_ens.npy') #ens has shape:=[time steps,system dimension,ensemble number]
a_mean=np.load('filtered_mean.npy')

plt.figure(figsize=(16,8))
# Start time and end time chosen to view a part of time series
t_start=0
t_stop=100
# component to view
comp_=37
time=ob_gap*np.arange(a_mean.shape[0])
#plt.plot(time[t_start:t_stop],a1_mean[t_start:t_stop,comp_],c='r',alpha=1)

plt.plot(time[t_start:t_stop],a_ens[t_start:t_stop,comp_],c='r',alpha=0.3)
plt.plot(time[t_start:t_stop],a_mean[t_start:t_stop,comp_],c='blue',alpha=1.0,label='Filter mean')
plt.plot(time[t_start:t_stop],state[t_start:t_stop,comp_],c='black',label='State')
if (comp_%2==0):
    #plt.scatter(time[t_start:t_stop],obs[t_start:t_stop,int(comp_/2)],c='black',edgecolors='black',marker='.',s=150,label='obs')
    plt.errorbar(x=time[t_start:t_stop],y=obs[t_start:t_stop,int(comp_/2)],yerr=mu,c='g',mec='black',mfc='g',capsize=4,fmt='.',ms=13)
plt.legend()
plt.title(r'An unobserved component'.format(comp_))
os.chdir('/home/shashank/Documents/enkf_for_clv2/plots/L63')
plt.savefig('example_enkf_uob.pdf')
plt.show()
#plt.xticks(time[t_start:t_stop],fontsize=12)
#plt.legend(frameon='True')
#plt.savefig('overfit_2.png')