import numpy as np

def sigmoid(x):
	try:
		return 1.0 / (1.0 +np.exp(-x))
	except:
		print("overflow:",x)
def sigmoidPrime(x):
	return sigmoid(x)*(1.0-sigmoid(x))


class MLP(object):

	def __init__(self,arch):
		self.ETA = 0.4
		self.numHiddenLayers = len(arch[1:-1])
		self.hiddenLayerSizes = [x for x in arch[1:-1]]#it'll have to be a list for many hidden layers
		self.inputSize = arch[0]
		self.outputSize = arch[-1]
		self.weights = [] # list of the weights matrices of each layer step
		self.biases = []
		self.tempW1 = 0
		self.tempW2 = 0
		self.tempW3 = 0
		#inicialize the weights
		for i in range(len(arch)-1):
			self.weights.append(np.random.randn(arch[i],arch[i+1]))#first weight matrix
			self.biases.append(np.random.randn(1,arch[i+1]))

	def biasMatrix(self,shape,biasVec):
		auxBiases = np.zeros(shape)

		for i in range(shape[0]):
			auxBiases[i] = auxBiases[i]+biasVec
		return auxBiases

	def feedForward(self,X):
		self.z2 = np.dot(X,self.weights[0]) + self.biases[0][0:X.shape[0]]
		self.a2 = sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.weights[1]) + self.biases[1][0:X.shape[0]]
		self.a3 = sigmoid(self.z3)
		self.z4 = np.dot(self.a3,self.weights[2])
		yhat = sigmoid(self.z3)#a3
		return yhat

	def backpropagation(self,X,Y,Yhat):

		delta4 = (Yhat - Y) * sigmoidPrime(self.z4)
		grad_Jw3 = np.dot(self.a3.T,delta4)

		delta3 = np.dot(delta4,self.weights[2]) * sigmoidPrime(self.z3)
		grad_Jw2 = np.dot(self.a2.T,delta3)

		delta2 = np.dot(delta3,self.weights[1].T) * sigmoidPrime(self.z2)
		grad_Jw1 = np.dot(X.T,delta2)

		return grad_Jw1,grad_Jw2,grad_Jw3,delta4,delta3,delta2

	def train(self,X,Y):
		#forward
		self.yhat = self.feedForward(X)
		#backpropagation

		self.grad_Jw1, self.grad_Jw2,self.grad_Jw3,self.delta4,self.delta3, self.delta2 = self.backpropagation(X,Y,self.yhat)
		# print(self.grad_Jw1)
		self.weights[0] += self.ETA * (-self.grad_Jw1) + 0.03 * (self.weights[0]-self.tempW1)
		self.weights[1] += self.ETA * (-self.grad_Jw2) + 0.03 * (self.weights[1]-self.tempW2)
		self.weights[2] += self.ETA * (-self.grad_Jw3) + 0.03 * (self.weights[2]-self.tempW3)
		self.tempW1 = self.weights[0]
		self.tempW2 = self.weights[1]
		self.tempW3 = self.weights[2]
		# print(self.biases[1].shape)
		# print(self.delta3.shape)
		self.biases[0] = self.biases[0] + (self.ETA *(-self.delta2))
		self.biases[1] = self.biases[1] + (self.ETA *(-self.delta3))
		# cost =self.cost(X,Y)
		# print(cost)
	def cost(self,X,Y):
		yhat = self.feedForward(X)
		J = 0.5 * sum( (Y - yhat)**2 )
		return J
def main():
	DATA = np.genfromtxt('wtrain.csv',delimiter=';')
	TEST = np.genfromtxt('wtest.csv',delimiter=';')

	num_samples = DATA.shape[0]
	print(num_samples)
	X = np.delete(DATA,11,1)#[0:100]
	Y = DATA.take(11,1).reshape((num_samples,1))#[0:100]
	Y = Y/10

	XT = np.delete(TEST,11,1)#[0:100]
	YT = TEST.take(11,1).reshape((400,1))#[0:100]
	YT = YT/10

	archMLP = [11,10,5,1]

	mlp = MLP(archMLP)

	#
	# X = np.array([	[0,0],
	# 	 			[0,1],
	# 	 			[1,0],
	# 	 			[1,1]])
	#
	# Y = np.array([[0,1,1,0]]).T
	np.seterr(over='raise')
	erroMin = 0.000005
	for i in range(0,100000):
		mlp.train(X,Y)
		if i %10 == 0:
			#print(mlp.feedForward(X))
			custo = mlp.cost(X,Y)
			print(custo)
			if(custo < erroMin):
				break

	a = mlp.feedForward(XT)
	print(a)
	a = np.round(10*a)
	acertos = 0

	for y,label in zip(a,10*YT):
		print(y ," ==? ",label,"acertos : ",acertos,"/",199)
		if y == label:
			acertos+=1


	np.savetxt('trainresult.out', a,delimiter=',')
	np.savetxt('pesos2.out', mlp.weights[0],delimiter=',')
	np.savetxt('pesos3.out', mlp.weights[1],delimiter=',')
	np.savetxt('b2.out', mlp.biases[0],delimiter=',')
	np.savetxt('b3.out', mlp.biases[1],delimiter=',')
	#mlp.think()
if __name__ == '__main__':
	main()
