The follow repository contains the ".py" files in the following structure
model-i is a place holder for different systems "L63" and "L9610", "L9620" and "L9640" repectively.

### The directory structure is as follows:

```bash ENKF_FOR_CLV2/  
├── README.md  
├── codes/  
│   └── model-i/  
│       ├── file1.py  
│       ├──  file2.py  
|       ├──  .....  
|       └──  fileN.py  
|  
├── data/  
|   └──model-i  
│       ├── file1.npy  
│       ├── file2.npy  
|       ├── .....  
|       └── fileN.npy  
|  
├── plots/  
│   └── model-i/  
│       ├──  img1.pdf  
│       ├──  img2.pdf  
|       ├──  .....  
|       └──  imgN.pdf  
```
## To perform the numerical experiments and produce the final plots for a specific model, we need to run the following set of scripts in sequence.

## Sensitivity of LVs using perturbed trajectories:
Run the following codes inside 'codes\model-i\' as follows:
1. Generate an initial condition on the attractor '\generate_initial_condition_on_att.py'
2. Generate a trajectory using this initial condition '\multiple_trajectories.py'
3. Generate noisy trajectories from this trajectroy '\generate_noisy_traj.py'
4. Perform calculations of LVs for all the generated trajectories using by running '\Computing_CLV_sensitivity.py'
5. Once obatined, compute the angles between the LVs from the noisy and the original trajectory using '\noisy_cosine_and_exp_calc_.py'
6. To finally plot the results, use '\final_plots_sensitivity.py'

The code folder contain 'assim' in their folder names.
## LVs using filter approximated trajectories:
Run the following codes inside 'codes\model-i\' as follows:
1. Generate an initial condition on the attractor '\generate_initial_condition_on_att.py'
2. Generate a trajectory using this initial condition '\multiple_trajectories.py'
3. Generate noisy ovservations from this trajectroy '\generate_obs.py'
4. Perform data assimilation using enkf for the generated observations using '\run_experiments_jaxed_enkf.py'
5. Perform calculations of LVs for all the filter obtained trajectories using by running '\Computing_CLV_sensitivity.py'
6. Once obatined, compute the angles between the LVs from the noisy and the original trajectory using '\noisy_cosine_and_exp_calc_.py'
7. To finally plot the results, use '\final_plots_sensitivity.py'
