import matplotlib.pyplot as plt
import os, sys
import cPickle

HOME = os.path.abspath(os.path.dirname(__file__))
fname = sys.argv[1]

ax = cPickle.load(open(HOME + '/figures/' + fname, 'r'))

plt.show()
