import numpy as np

def sign(x):
	return 1/(1+np.exp(x))
#perceptron de 2 entradas
data = np.loadtxt('prova3 percp.csv',delimiter=',', skiprows=1)

inputs = np.delete(data,2,1)
outputs = data.take(2,1).reshape((inputs.shape[0],1))

for i in range(len(outputs)):
	if(outputs[i] == 2):
		outputs[i] = -1
eta = 0.5

W = np.random.randn(2,1)
bw = np.random.rand()
b = -1
epochs = 300
#==========
for i in range(epochs):
	soma = inputs.dot(W)
	soma = soma + b
	ativ = np.sign(soma)
	#print(W)
	##erro
	err = outputs - ativ
	W = W + eta * np.dot(inputs.T,err)
	#b = b + eta * (sum(err)/32)
	b = b + eta*sum(err)
	#print(W)
	#print(ativ)
print("Pesos:",W,"\nbias: ",b)

print(ativ-outputs)








