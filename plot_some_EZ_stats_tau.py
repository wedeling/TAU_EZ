def get_pde(X, Npoints = 300):

    kernel = stats.gaussian_kde(X)
    x = np.linspace(np.min(X), np.max(X), Npoints)
    pde = kernel.evaluate(x)
    return x, pde

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from itertools import product, cycle, combinations
from scipy import stats
import sys
import json

HOME = os.path.abspath(os.path.dirname(__file__))

Omega = 7.292*10**-5
day = 24*60**2*Omega
sim_ID = 'tau_EZ_PE_HF_global'
t_end = (250.0 + 8*365.)*day 

N_inputs = 0
try:
    for i in range(1000):
        tmp = sys.argv[N_inputs + 1]
        N_inputs += 1
except IndexError:
    print 'There are', N_inputs, 'inputs'

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, xlabel=r'energy', yticks = [])
ax2 = fig.add_subplot(122, xlabel=r'enstropy', yticks = [])

fig2 = plt.figure('time_series')
ax3 = fig2.add_subplot(111)

#fig3 = plt.figure('rho')
#ax4 = fig3.add_subplot(111)

fig4 = plt.figure('tau')
ax5 = fig4.add_subplot(111, xlabel=r'$t\;[days]$')

markers = cycle(['r', 'g', 'b', 'y'])
lbl = cycle([r'$\mathrm{reduced}$']) 

for i in range(N_inputs):

    mark = markers.next()

    #####################################
    # read the inputs for the surrogate #
    #####################################
    fpath = sys.argv[i+1]
    fp = open('./inputs/' + fpath, 'r')
    N_surr = int(fp.readline())
    inputs = []
    
    for j in range(N_surr):
        inputs.append(json.loads(fp.readline()))

    fname = HOME + '/samples/' + sim_ID + '_' + fpath[0:-5] + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'

    print 'Loading samples ', fname

    try:
        #create HDF5 file
        h5f = h5py.File(fname, 'r')
         
        print h5f.keys()

        x_E_HF, pdf_E_HF = get_pde(h5f['e_n_HF'])
        x_E_LF, pdf_E_LF = get_pde(h5f['e_n_LF'])
        x_E_UP, pdf_E_UP = get_pde(h5f['e_n_UP'])
        x_Z_HF, pdf_Z_HF = get_pde(h5f['z_n_HF'])
        x_Z_LF, pdf_Z_LF = get_pde(h5f['z_n_LF'])
        x_Z_UP, pdf_Z_UP = get_pde(h5f['z_n_UP'])

        ax1.plot(x_E_LF, pdf_E_LF, mark, label=lbl.next() + r'$,\;\alpha = ' + str(inputs[0]['extrap_ratio'])+'$', alpha=0.5)
        ax2.plot(x_Z_LF, pdf_Z_LF, mark, label=lbl.next() + r'$,\;\alpha = ' + str(inputs[0]['extrap_ratio'])+'$', alpha=0.5)

        ax3.plot(h5f['t'], h5f['e_n_LF'])
        #ax4.plot(h5f['t'], h5f['rho'])
        ax5.plot(h5f['t'][:]/day, h5f['tau_E'],  label=r'$\tau_E$')
        ax5.plot(h5f['t'][:]/day, h5f['tau_Z'], '--', label=r'$\tau_Z$')
        #ax5.plot(h5f['t'][:]/day, h5f['r_tau_E'],  label=r'$\widetilde{\tau_E}$')
        #ax5.plot(h5f['t'][:]/day, h5f['r_tau_Z'], '--', label=r'$\widetilde{\tau_Z}$')

        print 'Mean tau_E =', np.mean(h5f['tau_E'])
        print 'Mean tau_Z =', np.mean(h5f['tau_Z'])
        #print 'Mean r_tau_E =', np.mean(h5f['r_tau_E'])
        #print 'Mean r_tau_Z =', np.mean(h5f['r_tau_Z'])

        if i == 0:
            ax1.plot(x_E_HF, pdf_E_HF, '--k', label=r'$\mathrm{reference}$')
            ax2.plot(x_Z_HF, pdf_Z_HF, '--k', label=r'$\mathrm{reference}$')
            ax1.plot(x_E_UP, pdf_E_UP, ':k', label=r'$\mathrm{unparam.}$')
            ax2.plot(x_Z_UP, pdf_Z_UP, ':k', label=r'$\mathrm{unparam.}$')
            plt.tight_layout()
            
            ax3.plot(h5f['t'], h5f['e_n_HF'], '--k')
            ax3.plot(h5f['t'], h5f['e_n_UP'], ':k')

    except IOError:
        print '*****************************'
        print fname, ' not found'
        print '*****************************'

handles,labels = ax1.get_legend_handles_labels()
#auto reorder
#handles = [handles[0], handles[2], handles[3], handles[4], handles[5], handles[1]]
#labels = [labels[0], labels[2], labels[3], labels[4], labels[5], labels[1]]
#Jac_LF reorder
#handles = [handles[0], handles[2], handles[3], handles[4], handles[1]]
#labels = [labels[0], labels[2], labels[3], labels[4], labels[1]]

#handles = [handles[1], handles[2], handles[0], handles[3], handles[4], handles[5], handles[6]]
#labels = [labels[1], labels[2], labels[0], labels[3], labels[4], labels[5], labels[6]]

ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
leg1 = ax1.legend(handles, labels, loc = 0)
leg2 = ax2.legend(handles, labels, loc = 0)
leg5 = ax5.legend()

leg1.draggable(True)
leg2.draggable(True)
leg5.draggable(True)

fig.tight_layout()

plt.show()
