# A random test
import numpy as np

seed_num=7
model_dim=5
noise_cov=np.eye(model_dim)
np.random.seed(seed_num)


noisy_=np.random.multivariate_normal(np.zeros(model_dim),noise_cov,3)
print(noisy_)

noisy_=np.random.multivariate_normal(np.zeros(model_dim),noise_cov,3)
print(noisy_)

noisy_=np.random.multivariate_normal(np.zeros(model_dim),noise_cov,3)
print(noisy_)