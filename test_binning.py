import numpy as np
import matplotlib.pyplot as plt
import sys, os, json, h5py
from itertools import product

plt.close('all')

HOME = os.path.abspath(os.path.dirname(__file__))

sim_ID = 'tau_EZ_PE_HF'
store_frame_rate = 1
sim_number = sys.argv[1]

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#start, end time (in days) + time step
t = 250.0*day
t_data = t + 8.0*365.0*day 

###########################
# load the reference data #
###########################
fname = HOME + '/samples/' + sim_ID + '_exact_training_t_' + str(np.around(t_data/day, 1)) + '.hdf5'

print 'Loading', fname

h5f = h5py.File(fname, 'r')

print h5f.keys()

#####################################
# read the inputs for the surrogate #
#####################################
fpath = sys.argv[2]
fp = open(fpath, 'r')
N_surr = int(fp.readline())
inputs = []

for i in range(N_surr):
    inputs.append(json.loads(fp.readline()))
    
print '****************************'
print 'Creating', N_surr, ' surrogates'
print '****************************'

#########################
# create the surrogates #
#########################

surrogate = {}; N_c = {}; covariates = {}; lags = {}; j3 = {}
for j in range(N_surr):

    param = inputs[j]
    
    #read the dict
    target = param['target']
    N_c[target] = param['N_c'] 
    covariates[target] = param['covariates']
    lag = param['lag']
    extrap_ratio = param['extrap_ratio']

    #reduce training data if extrap_ratio < 1 (to test extrapolative capability of surrogate)
    S_tot = h5f['t'].size
    S_train = np.int(extrap_ratio*h5f['t'].size)
    S_extrap = S_tot - S_train

    #spatially constant lag per covariate
    lags[target] = np.zeros(N_c[target]).astype('int')
    for i in range(N_c[target]):
        lags[target][i] = lag[i]
    
    max_lag = np.max(lags[target])
    min_lag = np.min(lags[target])
    j3[target] = 0#min_lag*store_frame_rate
    
    print '***********************'
    print 'Parameters'
    print '***********************'
    print 'Sim number =', sim_number
    print 'Target =', target
    print 'Covariates =', covariates[target]
    print 'Excluding', S_extrap, ' points from training set' 
    print 'Lags =', lags[target]
    print '***********************'

    #covariates
    c_i = np.zeros([S_train - max_lag, N_c[target]])
    i1 = max_lag; i2 = S_tot - S_extrap

    for i in range(N_c[target]):

        lag_i = lags[target][i] 

        if covariates[target][i] == 'auto' and target == 'dE':
            c_i[:, i] = h5f['e_n_HF'][i1-lag_i:i2-lag_i] - h5f['e_n_LF'][i1-lag_i:i2-lag_i]
        elif covariates[target][i] == 'auto' and target == 'dZ':
            c_i[:, i] = h5f['z_n_HF'][i1-lag_i:i2-lag_i] - h5f['z_n_LF'][i1-lag_i:i2-lag_i]
        elif covariates[target][i] == 'r_tau_E*sprime_n_LF':
            c_i[:, i] = h5f['tau_E'][i1-lag_i:i2-lag_i]*h5f['sprime_n_LF'][i1-lag_i:i2-lag_i]
        elif covariates[target][i] == 'r_tau_Z*zprime_n_LF':
            c_i[:, i] = h5f['tau_Z'][i1-lag_i:i2-lag_i]*h5f['zprime_n_LF'][i1-lag_i:i2-lag_i]
        else:
            c_i[:, i] = h5f[covariates[target][i]][i1-lag_i:i2-lag_i]

    if target == 'dZ':
        r = h5f['z_n_HF'][i1:i2] - h5f['z_n_LF'][i1:i2]
    elif target == 'dE':
        r = h5f['e_n_HF'][i1:i2] - h5f['e_n_LF'][i1:i2]
    
    #########################
    
    N_bins = 10

    print 'Creating Binning object...'
    from binning import *
    surrogate[target] = Binning(c_i, r.flatten(), 1, N_bins, lags = lags[target], store_frame_rate = store_frame_rate, verbose=True)
    #surrogate[target].plot_samples_per_bin()
    #if N_c == 1:
    #    surrogate[target].compute_surrogate_jump_probabilities(plot = True)
    #    surrogate[target].compute_jump_probabilities()
    #    surrogate[target].plot_jump_pmfs()
    print 'done'

    surrogate[target].print_bin_info()
    

surrogate[target].fill_in_blanks()
surrogate[target].plot_2D_binning_object()
print '--------------------'

ax = plt.gca()

#plot non empty midpoints
#ax.plot(surrogate[target].midpoints[:,0], surrogate[target].midpoints[:,1], 'g+')

#specify a deliberate outlier
c_i = np.array([0.000413, 0.0109]).reshape([1, 2])
c_i = np.array([0.000413, 0.109]).reshape([1, 2])
ax.plot(c_i[0][0], c_i[0][1], 'r+')

#evaluate surrogate at outlier
surrogate[target].get_r_ip1(c_i)

#check = 130
#x_idx_check = np.unravel_index(check, [len(b) + 1 for b in surrogate[target].bins])
#x_idx_check = [x_idx_check[i] - 1 for i in range(surrogate[target].N_c)]
#ax.plot(x_mid[0][x_idx_check[0]], x_mid[1][x_idx_check[1]], 'ro')

#plt.axis('equal')
plt.tight_layout()
plt.show()


