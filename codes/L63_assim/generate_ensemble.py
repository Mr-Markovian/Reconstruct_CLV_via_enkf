import numpy as np
import os
seed_num=2111997
np.random.seed(211197)

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63_assim'

os.chdir(data_path)
X_o=np.load('Initial_condition_on_att_L63.npy')

#initial bias in the ensemble........
biases=np.array([6.0])

#initial ensemble spread
covs_=np.array([2.0])
#Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])

# Finalized experiments in April,2021

#os.mkdir('ensembles')
#os.chdir(os.path.join(os.getcwd(),'ensembles'))

for i in range(len(biases)):
    bias=biases[i]
    for j in range(len(covs_)):
        "Select seeds such that there is a unique value for each set of complete parameters"
        seed_num=int(47*(i+1)*(j+1))
        np.random.seed(seed_num)
        mu=covs_[j]
        Initial_cov=mu*np.eye(3)
        x0_ensemble=np.random.multivariate_normal(X_o+bias,Initial_cov,50).T
        np.save('Ensembles={}_bias_{}_init_cov_{}_seed_{}.npy'.format(50,bias,covs_[j],seed_num),x0_ensemble)
        #np.save('Ensemble={}_init_cov_{}_seed_{}.npy'.format(i,parameters['lambda_'],seed_num),x0_ensemble)

print('Job Done')

