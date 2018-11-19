import numpy as np

a = np.genfromtxt('trainresult.out',delimiter=',')

a = np.round(a*10)
np.savetxt('trainresult2.out', a,delimiter=',')
