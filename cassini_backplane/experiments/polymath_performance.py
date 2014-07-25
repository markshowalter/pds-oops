from polymath import *
import numpy as np
import cProfile, pstats, StringIO

a = np.arange(1000*3*3.*3).reshape((1000,3,3,3))
a = np.rollaxis(a, 2, 4)

pr = cProfile.Profile()
pr.enable()
for i in range(1000):
    d = np.sum(a.copy(), axis=-1)
#    d = np.sum(a, axis=-1)
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
ps.print_callers()
print s.getvalue()
