import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, cPickle

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

sim_ID = 'tau_EZ_PE_HF'

Omega = 7.292*10**-5
day = 24*60**2*Omega
t_data = 500.0*day
#the dt of THE DATA
dt = 0.01

#open reference data
fname = HOME + '/samples/' + sim_ID + '_exact_training_t_' + str(np.around(t_data/day, 1)) + '.hdf5'

print 'Loading', fname

h5f = h5py.File(fname, 'r')
print h5f.keys()

S = h5f['t'].size

#number of lags to consider
max_lag = np.int(S/2.0)

lags = np.arange(1, max_lag)
R = np.zeros(lags.size)

idx = 0

QoI = 'dE'

#for every lag, compute autocorrelation:
# R = E[(X_t - mu_t)*(X_s - mu_s)]/(std_t*std_s)
for lag in lags:

    X_t = h5f[QoI][0:-lag]
    X_s = h5f[QoI][lag:]

    mu_t = np.mean(X_t)
    std_t = np.std(X_t)
    mu_s = np.mean(X_s)
    std_s = np.std(X_s)

    R[idx] = np.mean((X_t - mu_t)*(X_s - mu_s))/(std_t*std_s)
    idx += 1

cutoff = np.exp(1)**-1
idx_cutoff = np.where(R <= cutoff)[0][0]
tau_cutoff = idx_cutoff*dt/day

print 'Tau_cutoff = ', tau_cutoff

fig = plt.figure('acf_' + QoI)
ax = fig.add_subplot(111, xlabel=r'$\mathrm{lag\;[days]}$', ylabel=r'$\mathrm{autocorrelation\;function}$', xlim=[0, max_lag*dt/day])

ax.vlines(lags*dt/day, 0, R, colors='lightgray')
ax.vlines(tau_cutoff, 0, R[idx_cutoff], colors='darkgray', lw=4)

plt.tight_layout()

#store ax object to edit it later if needed
store_fig = True
if store_fig == True:
    ax = plt.gca()

    if os.path.exists(HOME + '/figures') == False:
        os.makedirs(HOME + '/figures')
    
    #generate random filename
    cPickle.dump(ax, open(HOME + '/figures/acf_' + QoI + '.pickle', 'w'))

#matplotlib equivalent
fig = plt.figure()
ax = fig.add_subplot(111)
ax.acorr(h5f[QoI], maxlags=max_lag)
plt.tight_layout()

plt.show()
