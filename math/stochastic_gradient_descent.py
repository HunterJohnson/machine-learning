import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
alpha,hidden_dim = (0.5,4)
synapse_0 = 2*np.random.random((3,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1
 
for j in range(60000):
	for i in range(4):	# Nested Loop to process each sample individually
	    layer_1 = 1/(1+np.exp(-(np.dot(X[i], synapse_0))))
	    layer_2 = 1/(1+np.exp(-(np.dot(layer_1, synapse_1))))
	    layer_2_delta = (layer_2 - y[i][0])*(layer_2*(1-layer_2))
	    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
	    synapse_1 -= [[alpha*x*layer_2_delta[0]] for x in layer_1]
	    synapse_0 -=  alpha * np.array([[x] for x in X[i]]).dot(np.array([[x] for x in layer_1_delta]).T)
