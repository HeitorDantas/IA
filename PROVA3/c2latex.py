import numpy as np

def c2l(csvf,latexf):
	l = []
	csv = open(csvf,'r')
	lat = open(latexf,'w')

	lines = csv.readlines()
	lat.write("\\begin{bmatrix}\n")
	for l in lines:
		rl = l.replace(',','&')
		rl = rl.replace('\n','\\\\\n')
		lat.write(rl)

	lat.write("\\end{bmatrix}")

	csv.close()
	lat.close()
#----------------------------------------
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def sigmoidPrime(x):
	return sigmoid(x) * (1-sigmoid(x))
#-----------------------------------------

#c2l('prova3.csv','teste.out')
path = "./Arquivos/"
'''
W1 = np.random.randn(2,2)
W2 = np.random.randn(2,3)
B2 = np.random.randn(2,1)
B3 = np.random.randn(3,1)

np.savetxt(path+"w1.out",W1,delimiter=',',fmt='%.2f')
np.savetxt(path+"w2.out",W2,delimiter=',',fmt='%.2f')
np.savetxt(path+"b2.out",B2,delimiter=',',fmt='%.2f')
np.savetxt(path+"b3.out",B3,delimiter=',',fmt='%.2f')
'''
X = np.loadtxt("prova3.csv",delimiter=',',skiprows=1)
Y = X.take(2,1).reshape(32,1)
X = np.delete(X,2,1)

W1 = np.loadtxt(path+'w1.out',delimiter=',')
W2 = np.loadtxt(path+'w2.out',delimiter=',')
B2 = np.loadtxt(path+'b2.out',delimiter=',')
B3 = np.loadtxt(path+'b3.out',delimiter=',')
'''
#gravar em latex
c2l(path+'w1.out',path+'w1.tex')
c2l(path+'w2.out',path+'w2.tex')
c2l(path+'b2.out',path+'b2.tex')
c2l(path+'b3.out',path+'b3.tex')
'''
#=======feedforward===========

z2 = np.dot(X,W1) + B2
a2 = sigmoid(z2)
z3 = np.dot(a2,W2) + B3
a3 = sigmoid(z3)#saida
'''
#fluxos
np.savetxt(path+"z2.out",z2,delimiter=',',fmt='%.2f')
np.savetxt(path+"a2.out",a2,delimiter=',',fmt='%.2f')
np.savetxt(path+"z3.out",z3,delimiter=',',fmt='%.2f')
np.savetxt(path+"a3.out",a3,delimiter=',',fmt='%.2f')

c2l(path+'z2.out',path+'z2.tex')
c2l(path+'a2.out',path+'a2.tex')
c2l(path+'z3.out',path+'z3.tex')
c2l(path+'a3.out',path+'a3.tex')
#==========Backpropagtion=======
'''
delta3 = (Y - a3) *sigmoidPrime(z3)
gradw2 = np.dot(a2.T,delta3)
delta2 = np.dot(delta3,W2.T) * sigmoidPrime(z2)
gradw1 = np.dot(X.T,delta2)
'''
#fluxos
np.savetxt(path+"gradb3.out",delta3,delimiter=',',fmt='%.2f')
np.savetxt(path+"gradw2.out",gradw2,delimiter=',',fmt='%.2f')
np.savetxt(path+"gradb2.out",delta2,delimiter=',',fmt='%.2f')
np.savetxt(path+"gradw1.out",gradw1,delimiter=',',fmt='%.2f')

c2l(path+'gradw1.out',path+'gradw1.tex')
c2l(path+'gradw2.out',path+'gradw2.tex')
c2l(path+'gradb2.out',path+'gradb2.tex')
c2l(path+'gradb3.out',path+'gradb3.tex')
#====================================
'''
#atualizar pesos
eta = 0.8
nW1 = W1 - eta*gradw1
nW2 = W2 - eta * gradw2
nB2 = B2 - eta * delta2
nB3 = B3 - eta * delta3

print(B3)
print(delta3)
print(B3-eta*delta3)


#fluxos
np.savetxt(path+"nb3.out",nB3,delimiter=',',fmt='%.2f')
np.savetxt(path+"nw2.out",nW2,delimiter=',',fmt='%.2f')
np.savetxt(path+"nb2.out",nB2,delimiter=',',fmt='%.2f')
np.savetxt(path+"nw1.out",nW1,delimiter=',',fmt='%.2f')

c2l(path+'nw1.out',path+'nw1.tex')
c2l(path+'nw2.out',path+'nw2.tex')
c2l(path+'nb2.out',path+'nb2.tex')
c2l(path+'nb3.out',path+'nb3.tex')

