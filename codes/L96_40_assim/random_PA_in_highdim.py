# Principal angles in high-dimensions for random vectors, we need to generate statistics
# for the angles computed from a finite number of such randomly generated pairs.

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles,svdvals



#ambient dimension
model_dim=40
subspace_dim=20

num_realizations=1000
c=np.zeros((num_realizations,subspace_dim))

data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}/Sensitivity/PA'.format(model_dim)
os.chdir(data_path)
#If A and B are generated using uniform distribution:
#A= np.random.uniform(-1.0,1.0,size=subspace_dim*dim).reshape((dim,subspace_dim))
#B= np.random.uniform(-1.0,1.0,size=subspace_dim*dim).reshape((dim,subspace_dim))

for i in range(num_realizations):
    A= np.random.normal(size=subspace_dim*model_dim).reshape((model_dim,subspace_dim))
    #B= np.random.normal(size=subspace_dim*dim).reshape((dim,subspace_dim))
    B=np.eye(40)[:,:subspace_dim]
    A_q,_=np.linalg.qr(A)
    B_q,_=np.linalg.qr(B)
    c[i]=np.rad2deg(np.flip(subspace_angles(A_q,B_q)))

np.save('random_subspaces_PA_model_dim={}_subspaces_={}.npy'.format(model_dim,subspace_dim),c)
plt.show()

