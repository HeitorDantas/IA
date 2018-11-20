import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 +np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x)*(1.0-sigmoid(x))


class MLP(object):

	def __init__(self,arch):
		self.ETA = 0.2
		self.numHiddenLayers = len(arch[1:-1])
		self.hiddenLayerSizes = [x for x in arch[1:-1]]#it'll have to be a list for many hidden layers
		self.inputSize = arch[0]
		self.outputSize = arch[-1]
		self.weights = [] # list of the weights matrices of each layer step
		self.biases = []
		self.tempW1 = 0
		self.tempW2 = 0
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
		# a = X
		# activations = [X]
		# print(self.biases[0].shape)

		self.z2 = np.dot(X,self.weights[0]) + self.biases[0]
		# auxBiases = self.biasMatrix(self.z2.shape,self.biases[0])
		# self.z2 = self.z2 + auxBiases
		# print(self.z2.shape)
		self.a2 = sigmoid(self.z2)

		self.z3 = np.dot(self.a2,self.weights[1]) + self.biases[1]
		# auxBiases = self.biasMatrix(self.z3.shape,self.biases[1])
		# self.z3 = self.z3 + auxBiases
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

		self.weights[0] += self.ETA * (-self.grad_Jw1) + 0.03 * self.tempW1
		self.weights[1] += self.ETA * (-self.grad_Jw2) + 0.03 * self.tempW2
		self.tempW1 = self.grad_Jw1
		self.tempW2 = self.grad_Jw2
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
	DATA = np.genfromtxt('winequality-red.csv',delimiter=';')
	

	p2 = np.genfromtxt('pesos2.out',delimiter=',')
	p3 = np.genfromtxt('pesos3.out',delimiter=',')
	b2 = np.genfromtxt('b2.out',delimiter=',')
	b3 = np.genfromtxt('b3.out',delimiter=',')

	num_samples = DATA.shape[0]
	print(num_samples)
	X = np.delete(DATA,11,1)#[0:100]
	Y = DATA.take(11,1).reshape((num_samples,1))#[0:100]
	Y = Y/10

	archMLP = [11,5,1]

	mlp = MLP(archMLP)
	mlp.weights[0] = p2
	mlp.weights[1] = p3
	mlp.biases[0] = b2[0]
	mlp.biases[1] = b3[0]

	print(mlp.feedForward(X)[0:100])

if __name__ == '__main__':
	main()
