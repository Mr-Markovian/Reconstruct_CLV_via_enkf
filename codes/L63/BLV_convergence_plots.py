# Convergence plots
"""To see if the BLVs converge to the same values starting with two different points on the trajectory"""
import numpy as np
import os
import matplotlib.pyplot as plt

dt=0.01  
dt_solver=0.002
model_dim=3
model_name='L63'
num_clv=3
seed_num=40
n1=10000
qrstep=10

spr=1
data_path='/home/shashank/Documents/enkf_for_clv2/data/L63/Sensitivity'
os.chdir(data_path)

#os.chdir('L96_BLV_test2')

# Load trajectories which start from a fixed initial point
# ------------------------- Test using the code for BLV from a point with qr and p---------
# G=np.load('BLV_qr{}_p{}.npy'.format(qrstep,0))

# startpt=100
# G1=np.load('BLV_qr{}_p{}.npy'.format(qrstep,startpt))

# # Angle between BLVs
# cosines=np.zeros((int(n1/qrstep),num_clv))
# for i in range(int(n1/qrstep)):
#     for j in range(num_clv):
#         cosines[i,j]=np.absolute(np.dot(G1[i,:,j],G[i,:,j]))

# # Convergence along the same trajectory         
# plt.figure(figsize=(12,8))
# for i in range(num_clv):
#     plt.plot(np.arange(0,len(cosines[:,0]))*qrstep*dt,cosines[:,i],label='{}'.format(i+1))
# plt.legend()
# plt.savefig('convergence_qr{}_p{}.pdf'.format(qrstep,startpt))
# #plt.savefig('convergence_evolve_dont_evolve_.png')
# plt.show()

# ----------------Test using the code for BLV about a trajectory with qr and p--------------

os.chdir('L63_BLV_test1')

G=np.load('BLV_qr{}_p{}_mtp.npy'.format(qrstep,0))

startpt=200
G1=np.load('BLV_qr{}_p{}_mtp.npy'.format(qrstep,startpt))

# Angle between BLVs
cosines=np.zeros((int(n1/qrstep),num_clv))
for i in range(int(n1/qrstep)):
    for j in range(num_clv):
        cosines[i,j]=np.absolute(np.dot(G1[i,:,j],G[i,:,j]))

# Convergence along the same trajectory         
plt.figure(figsize=(12,8))
for i in range(num_clv):
    plt.semilogy(np.arange(0,len(cosines[:500,0]))*qrstep*dt,cosines[:500,i],label='{}'.format(i+1))
plt.legend()
plt.savefig('convergence_qr{}_p{}.pdf'.format(qrstep,startpt))
#plt.savefig('convergence_evolve_dont_evolve_.png')
plt.show()

# b1 = G1
# b2 = G
# bd = b1 - b2
# nd = np.zeros([10000, 5])
# for ii in range(5):
#     for jj in range(G.shape[0]):
#         nd[jj,ii] = np.linalg.norm(bd[jj,:,ii])
# plt.figure(figsize=(10,5))
# plt.plot(nd[:,ii])
# plt.show()



