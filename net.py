import random, math
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv',header=0)

Y = np.array(df["label"])
X = np.array(df[["grade1","grade2"]])


def CleanLabels(label):
	if label == 0:
		return [0.0,1.0]
	elif label == 1:
		return [1.0,0.0]

Y = map(CleanLabels,Y)



class Network(object):

	def __init__(self):
		self.w1 = random.random()
		self.w2 = random.random()
		self.w3 = random.random()
		self.w4 = random.random()
		self.w5 = random.random()
		self.w6 = random.random()
		self.w7 = random.random()
		self.w8 = random.random()
		self.b1 = random.random()
		self.b2 = random.random()

	def SumOfSqErrors(self,X,Y):
		error = 0
		for i in xrange(len(X)):
			target = Y[i]
			inputs = X[i]
			output = self.Forward(target,inputs)
			error += self.SqErrors(target,output)
		return error / len(X)

	def SqErrors(self,target,output):
		error = 0
		for i in xrange(len(target)):
			error += 0.5*(target[i] - output[i])**2
		return error

	def SqErrorsDerivative(self,target,output):
		result = output - target
		return result
 
	def Error(self,target,output):
		error = 0.5*(target-output)**2
		return error

	def Sigmoid(self,z):
		return 1.0/ (1.0 + math.exp(-z) )

	def SigmoidDerivative(self,z):
		s = self.Sigmoid(z)
		return s*(1.0-s)

	def Forward(self,target,inputs):
		self.target = target
		self.inputs = inputs

		self.i1 = inputs[0]
		self.i2 = inputs[1]

		self.net_h1 = self.w1*self.i1 + self.w2*self.i2 + self.b1
		self.net_h2 = self.w3*self.i1 + self.w4*self.i2 + self.b1
		
		self.out_h1 = self.Sigmoid(self.net_h1)
		self.out_h2 = self.Sigmoid(self.net_h2)

		self.net_o1 = self.w5*self.out_h1 + self.w6*self.out_h2 + self.b2
		self.net_o2 = self.w7*self.out_h1 + self.w8*self.out_h2 + self.b2

		self.out_o1 = self.Sigmoid(self.net_o1)
		self.out_o2 = self.Sigmoid(self.net_o2)

		#error = self.Error(self.out_o1,target[0]) + self.Error(self.out_o1,target[1])
		self.output = [self.out_o1,self.out_o2]

		return [self.out_o1,self.out_o2]

	def Backward(self):
		dE1_WRT_out_o1 = self.SqErrorsDerivative(self.target[0],self.output[0])
		dE2_WRT_out_o2 = self.SqErrorsDerivative(self.target[1],self.output[1])

		dOut_o1_WRT_net_o1 = self.SigmoidDerivative(self.net_o1)
		dOut_o2_WRT_net_o2 = self.SigmoidDerivative(self.net_o2)

		dOut_h1_WRT_net_h1 = self.SigmoidDerivative(self.net_h1)
		dOut_h2_WRT_net_h2 = self.SigmoidDerivative(self.net_h2)

		dNet_h1_WRT_w1 = self.i1
		dNet_h1_WRT_w2 = self.i2
		dNet_h2_WRT_w3 = self.i1
		dNet_h2_WRT_w4 = self.i2

		dNet_o1_WRT_out_h1 = self.w5
		dNet_o1_WRT_out_h2 = self.w6
		dNet_o2_WRT_out_h1 = self.w7
		dNet_o2_WRT_out_h2 = self.w8

		# derivative of ERRORS with respect to h1
		dE1_WRT_out_h1 = dE1_WRT_out_o1 * dOut_o1_WRT_net_o1 * dNet_o1_WRT_out_h1
		dE2_WRT_out_h1 = dE2_WRT_out_o2 * dOut_o2_WRT_net_o2 * dNet_o2_WRT_out_h1
		dEtot_WRT_out_h1 = dE1_WRT_out_h1 + dE2_WRT_out_h1

		# derivative of ERRORS with respect to h2
		dE1_WRT_out_h2 = dE1_WRT_out_o1 * dOut_o1_WRT_net_o1 * dNet_o1_WRT_out_h2
		dE2_WRT_out_h2 = dE2_WRT_out_o2 * dOut_o2_WRT_net_o2 * dNet_o2_WRT_out_h2
		dEtot_WRT_out_h2 = dE1_WRT_out_h2 + dE2_WRT_out_h1

		dw1 = dEtot_WRT_out_h1 * dOut_h1_WRT_net_h1 * dNet_h1_WRT_w1
		dw2 = dEtot_WRT_out_h1 * dOut_h1_WRT_net_h1 * dNet_h1_WRT_w2
		dw3 = dEtot_WRT_out_h2 * dOut_h2_WRT_net_h2 * dNet_h2_WRT_w3
		dw4 = dEtot_WRT_out_h2 * dOut_h2_WRT_net_h2 * dNet_h2_WRT_w4

		dw5 = dE1_WRT_out_o1 * dOut_o1_WRT_net_o1 * self.out_h1
		dw6 = dE1_WRT_out_o1 * dOut_o1_WRT_net_o1 * self.out_h2
		dw7 = dE2_WRT_out_o2 * dOut_o2_WRT_net_o2 * self.out_h1
		dw8 = dE2_WRT_out_o2 * dOut_o2_WRT_net_o2 * self.out_h2

		alpha = 0.1
		self.w1 -= dw1 * alpha
		self.w2 -= dw2 * alpha
		self.w3 -= dw3 * alpha
		self.w4 -= dw4 * alpha
		self.w5 -= dw5 * alpha
		self.w6 -= dw6 * alpha
		self.w7 -= dw7 * alpha
		self.w8 -= dw8 * alpha


PerNet = Network()

def TestNetwork(Net,X,Y):
	score = 0
	for j in xrange(len(Y)):
		inputs = X[j]
		target = Y[j]
		prediction = Net.Forward(target,inputs)
		rounded_prediction = [round(prediction[0]),round(prediction[1])]
		print '---'
		print 'prediction: ', rounded_prediction
		print 'target    : ', target
		if rounded_prediction == target:
			score += 1


	print 'score is :', float(score) / 30.0
	print 'score is :', score



# train the network
def TrainNetwork(Net,iterations,X,Y,TrainSize):
	X_train = X[:TrainSize]
	Y_train = Y[:TrainSize]

	X_test = X[TrainSize:]
	Y_test = Y[TrainSize:]

	for x in xrange(iterations):
		i = random.randint(0,len(Y_train)-1)
		inputs = X_train[i]
		target = Y_train[i]
		output = Net.Forward(target,inputs)
		Net.Backward()
		error = Net.SumOfSqErrors(X_train,Y_train)
		if x % 1000 == 0:
			print error

	TestNetwork(Net,X_test,Y_test)

TrainNetwork(PerNet,10000,X,Y,70)






#print 'outputs before: ', outputs
#Net.Backward(targets,outputs)
#outputs = Net.Forward()
#print 'outputs after: ', outputs


#print Net.net_o1
