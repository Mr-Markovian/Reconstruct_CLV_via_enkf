from testbed_models import L63
import os
import numpy as np

seed_nums=np.array([13,17,19,23,29])

# these parameters now give you the mode, which is only a function of x, the state vector
model=L63
model_dim=3

#Observation Operator and noise statistics need to be defined first:
mu=0.7       
m=1   
n=model_dim
ob_gap=0.01

#Observation Operator for observing y 
H1_=np.zeros((m,n))
H1_[0,m]=1          # for y, choose the location for
obs_cov1=mu*np.eye(m)


data_path='/home/shashank/Documents/enkf_for_clv2/data/L63_assim'
os.chdir(data_path)

#Load the trajectory 
State=np.load('Multiple_trajectories_N=1_gap=0.01_ti=0.0_tf=350.0_dt_0.01_dt_solver=0.002.npy')
mus=np.array([0.1,0.3,0.5,0.7,0.9])

for j in range(5):
    os.chdir(data_path)
    os.mkdir('ob{}'.format(j+1))
    os.chdir(data_path+'/ob{}'.format(j+1))
    for i in range(5):
        mu=mus[i]
        obs_cov1=mu*np.eye(m)
        np.random.seed(seed_nums[i])
        obs=(H1_@(State.T)).T+np.random.multivariate_normal(np.zeros(m),obs_cov1,State.shape[0])
        np.save('ob{}_gap_{}_H1_'.format(j+1,round(ob_gap,2))+'_mu={}'.format(mu)+'_obs_cov1.npy',obs)

print(State.shape)
print(obs.shape)
print('Job Done')

