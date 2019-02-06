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


HOME = os.path.abspath(os.path.dirname(__file__))

Omega = 7.292*10**-5
day = 24*60**2*Omega
t_end = 500.0*day
t_end = (250.0 + 10.0*365)*day

sim_ID = 'RST_TEST5'
binning_type = 'global'
run = ''

N_inputs = 0
try:
    for i in range(1000):
        tmp = sys.argv[N_inputs + 1]
        N_inputs += 1
except IndexError:
    print 'There are', N_inputs, 'inputs'

fig = plt.figure(figsize=[5,7])
ax1 = fig.add_subplot(211, xlabel=r'energy', yticks = [])
ax2 = fig.add_subplot(212, xlabel=r'enstropy', yticks = [])

fig2 = plt.figure('time_series')
ax3 = fig2.add_subplot(111)

fig3 = plt.figure('rho')
ax4 = fig3.add_subplot(111)

fig4 = plt.figure('tau')
ax5 = fig4.add_subplot(111, xlabel=r'$t\;[days]$')

markers = cycle(['r', 'g', 'b', 'y'])
lbl = cycle(['with tau']) 

for i in range(N_inputs):

    mark = markers.next()

    fname = HOME + '/samples/' + sys.argv[i+1]

    print fname
    print 'Loading samples ', fname

    try:
        #create HDF5 file
        h5f = h5py.File(fname, 'r')
         
        print h5f.keys()

        x_E_HF, pdf_E_HF = get_pde(h5f['e_np1_HF'])
        x_E_LF, pdf_E_LF = get_pde(h5f['e_np1_LF'])
        #x_E_UP, pdf_E_UP = get_pde(h5f['e_UP'])
        x_Z_HF, pdf_Z_HF = get_pde(h5f['z_np1_HF'])
        x_Z_LF, pdf_Z_LF = get_pde(h5f['z_np1_LF'])
        #x_Z_UP, pdf_Z_UP = get_pde(h5f['z_UP'])

        ax1.plot(x_E_LF, pdf_E_LF, mark, label=lbl.next())
        ax2.plot(x_Z_LF, pdf_Z_LF, mark, label=lbl.next())

        ax3.plot(h5f['t'], h5f['e_np1_LF'])
        #ax4.plot(h5f['t'], h5f['rho'])
        ax5.plot(h5f['t'][:]/day, h5f['tau_E'],  label=r'$\tau_E$')
        ax5.plot(h5f['t'][:]/day, h5f['tau_Z'], '--', label=r'$\tau_Z$')

        print 'Mean tau_E =', np.mean(h5f['tau_E'])
        print 'Mean tau_Z =', np.mean(h5f['tau_Z'])

        if i == 0:
            ax1.plot(x_E_HF, pdf_E_HF, '--k', label=r'$\mathrm{reference}$')
            ax2.plot(x_Z_HF, pdf_Z_HF, '--k', label=r'$\mathrm{reference}$')
            #ax1.plot(x_E_UP, pdf_E_UP, ':k', label=r'$\mathrm{unparam.}$')
            #ax2.plot(x_Z_UP, pdf_Z_UP, ':k', label=r'$\mathrm{unparam.}$')
            ax3.plot(h5f['t'], h5f['e_np1_HF'], '--k')
            #ax3.plot(h5f['t'], h5f['e_UP'], ':k')

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

ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
leg1 = ax1.legend(handles, labels, loc = 0)
leg2 = ax2.legend(handles, labels, loc = 0)
leg5 = ax5.legend()

leg1.draggable(True)
leg2.draggable(True)
leg5.draggable(True)

plt.tight_layout()
plt.show()
