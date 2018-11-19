import numpy as np

def sigmoid(x):
	return 1 / (1 +np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x)*(1-sigmoid(x))

class MLP(object):
	def __init__(self,arch):
		self.ETA = 0.5
		self.numHiddenLayers = len(arch[1:-1])
		self.hiddenLayerSizes = [x for x in arch[1:-1]]#it'll have to be a list for many hidden layers
		self.inputSize = arch[0]
		self.outputSize = arch[-1]
		self.weights = [] # list of the weights matrices of each layer step
		self.biases = []
		#inicialize the weights
		for i in range(len(arch)-1):
			self.weights.append(np.random.randn(arch[i],arch[i+1]))#first weight matrix
			self.biases.append(np.random.randn(1,arch[i+1]))
		# for w,b in zip(self.weights,self.biases):
		# 	print(w)
		# 	print(b)
	def biasMatrix(self,shape,biasVec):
		auxBiases = np.zeros(shape)

		for i in range(shape[0]):
			auxBiases[i] = auxBiases[i]+biasVec
		return auxBiases

	def feedForward(self,X):
		a = X
		activations = [X]

		self.z2 = np.dot(X,self.weights[0]) + self.biases[0]
		auxBiases = self.biasMatrix(self.z2.shape,self.biases[0])
		self.z2 = self.z2 + auxBiases

		self.a2 = sigmoid(self.z2)

		self.z3 = np.dot(self.a2,self.weights[1])
		auxBiases = self.biasMatrix(self.z3.shape,self.biases[1])

		self.z3 = self.z3 + auxBiases
		yhat = sigmoid(self.z3)#a3
		return yhat

	def backpropagation(self,X,Y,Yhat):
		delta3 = (Yhat - Y) * sigmoidPrime(self.z3)
		grad_Jw2 = np.dot(self.a2.T,delta3)

		delta2 = np.dot(delta3,self.weights[1].T)* sigmoidPrime(self.z2)
		grad_Jw1 = np.dot(X.T,delta2)

		return grad_Jw1,grad_Jw2,delta3,delta2

	def train(self,X,Y):
		#forward
		self.yhat = self.feedForward(X)
		#backpropagation
		self.grad_Jw1, self.grad_Jw2,self.delta3, self.delta2 = self.backpropagation(X,Y,self.yhat)
		self.weights[0] += self.ETA * (-self.grad_Jw1)
		self.weights[1] += self.ETA * (-self.grad_Jw2)
		cost =self.cost(X,Y)
		#print(cost)
	def cost(self,X,Y):
		yhat = self.feedForward(X)
		J = 0.5 * sum( (Y - yhat)**2 )
		return J
	def costPrime(self):
		pass
def main():
	DATA = np.genfromtxt('winequality-red.csv',delimiter=';')
	num_samples = DATA.shape[0]

	X = np.delete(DATA,11,1)#[0:10000]
	Y = DATA.take(11,1).reshape((num_samples,1))#[0:10000]
	Y = Y / 10
	print(Y)
	archMLP = [11,15,1]

	mlp = MLP(archMLP)


	# X = np.array([	[0,0],
	# 	 			[0,1],
	# 	 			[1,0],
	# 	 			[1,1]])
	#
	# Y = np.array([[0,1,1,0]]).T

	for i in range(0,10000):
		mlp.train(X,Y)
		if i %10 == 0:
			print(mlp.feedForward(X))

	#mlp.think()
if __name__ == '__main__':
	main()
