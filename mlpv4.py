import numpy as np
#
def sigmoid(x):
	return 1 / (1 +np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x)*(1-sigmoid(x))

#
# def sigmoid(x):
# 	return x / (1 +abs(x))
# def sigmoidPrime(x):
# 	return 1/((1-abs(x))**2)

class MLP(object):
	def __init__(self,arch):
		self.ETA = 0.5
		self.numLayers = len(arch)
		self.weights = [] # list of the weights matrices of each layer step
		self.biases = []
		#inicialize the weights
		for i in range(len(arch)-1):
			self.weights.append(np.random.randn(arch[i],arch[i+1]))#first weight matrix
			self.biases.append(np.random.randn(1,arch[i+1]))

	def feedForward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(a,w)+b)
		return a
	def backpropagation(self,X,Y):
		a = X
		activations = [X]
		zs = []
		for i in range(len(self.weights)):
			z = np.dot(a,self.weights[i]) + self.biases[i]
			a = sigmoid(z)
			zs.append(z)
			activations.append(a)
		grads_b = [np.zeros(b.shape) for b in self.biases]
		grads_w = [np.zeros(w.shape) for w in self.weights]
		delta = (activations[-1] - Y) * sigmoidPrime(zs[-1])
		grads_b[-1] = delta
		grad = np.dot(activations[-2].T,delta)
		grads_w[-1] = grad

		for layer in range(2,self.numLayers):
			delta = np.dot(delta,self.weights[-layer+1].T) * sigmoidPrime(zs[-layer])
			grad = np.dot(activations[-layer-1].T,delta)
			grads_w[-layer] = grad
			grads_b[-layer] = delta
		return (grads_w,grads_b)

	def train(self,trainingData,epochs,eta,mini_batch_size):
		#grads_b = [np.zeros(b.shape) for b in self.biases]
		#grads_w = [np.zeros(w.shape) for w in self.weights]
		X = trainingData[0]
		Y = trainingData[1]
		size = X.shape[0]
		print(size)
		mini_batchesX = [X[k:k+mini_batch_size] for k in range(0, size, mini_batch_size)]
		mini_batchesY = [Y[k:k+mini_batch_size] for k in range(0, size, mini_batch_size)]
		for i in range(epochs):
			for mX,mY in zip(mini_batchesX,mini_batchesY):
				d_grads_w,d_grads_b = self.backpropagation(mX,mY)
				self.weights = [w-eta*nw for w, nw in zip(self.weights, d_grads_w)]
				self.biases =  [b-eta*nb for b, nb in zip(self.biases, d_grads_b)]
			#for x,y in zip(trainingData[0],trainingData[1]):
			# i = np.random.choice(range(size))
			# x = X[i]
			# y = Y[i]
			# d_grads_w,d_grads_b = self.backpropagation(x.reshape((1,116)),y)
			# grads_b = [nb+dnb for nb, dnb in zip(grads_b, d_grads_b)]
			# grads_w = [nw+dnw for nw, dnw in zip(grads_w, d_grads_w)]
			# self.weights = [w-eta*nw for w, nw in zip(self.weights, grads_w)]
			# self.biases =  [b-eta*nb for b, nb in zip(self.biases, grads_b)]

			# cost =self.cost(X,Y)
			# print(cost)
			if(i % 10 == 0):
				print(self.cost(mX,mY))
				#print(self.feedForward(X))

	def cost(self,X,Y):
		yhat = self.feedForward(X)
		J = 0.5 * sum( (Y - yhat)**2 )
		return J

def main():
	DATA = np.genfromtxt('dota2Train.csv',delimiter=',')
	num_samples = DATA.shape[0]

	X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
	Y = np.array([[0,1,1,0]]).T
	X2 = np.delete(DATA,0,1)#[0:10000]
	Y2 = DATA.take(0,1).reshape((num_samples,1))#[0:10000]
	for i in range(len(Y2)):
		if(Y2[i] == -1):
			Y2[i]=0
	#print(Y2[0:100])
	trainingData = [X2,Y2]
	#trainingData = zip(X,Y)
	print(X2.shape)
	print(Y2.shape)
	archMLP = [116,20,20,1]
	#archMLP = [2,3,1]
	mlp = MLP(archMLP)
	mlp.train(trainingData,100000,0.5,850)
	#mlp.feedForward(X)
	#mlp.think()
if __name__ == '__main__':
	main()
