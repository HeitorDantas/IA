import numpy as np

np.seterr(over='raise')
def sigmoid(x):
	try:
		return 1.0 / (1.0 +np.exp(-x))
	except:
		print("overflow:",x)


print(sigmoid(-7000))
