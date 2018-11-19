import numpy as np

a = np.random.randn(10,10)

np.savetxt('saida.out', a,delimiter=',')
