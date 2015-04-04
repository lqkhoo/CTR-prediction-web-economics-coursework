import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import time
import pickle as pickle
import csv

def load_data():
	xs_train = []
	with open('xs_train.dat', 'rb') as infile:
	    xs = pickle.load(infile)
	ys_train = np.load('ys_train.npy')
	xs_test = []

	with open('xs_test.dat', 'rb') as infile:
	    xs_test = pickle.load(infile)

	print("Loaded files")

	return xs_train, ys_train, xs_test


start = time.time()

np.set_printoptions(threshold=np.nan)

xs_train, ys_train, xs_test = load_data()

# Initialize linear model
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=0.001)

# Split input for 5 fold cross validation
cv = cross_validation.KFold(len(ys_train), 2, shuffle = False)

mses = []

for traincv, testcv in cv:
	X_train = xs_train[traincv]
	y_train = ys_train[traincv]

	X_test = xs_train[testcv]
	y_test = ys_train[testcv]

	clf.fit(X_train, y_train)
	p = clf.predict(X_test)

	'''
	d = clf.decision_function(X_test)
	print(p[1000])
	print(d[1000])
	print(1/(1 + np.exp(-d[1000])))
	'''

	mse = ((p - y_test) ** 2).mean()
	#s = clf.score(X_test, y_test)
	print("mse: ", mse)
	mses.append(mse)


mses = np.array(mses)
print("\n\nAVG MSE: ", np.mean(mses))

p = clf.predict_proba(xs_test)
print(clf.classes_)
n = len(p)

i = np.arange(1, n + 1, 1, dtype='int').reshape(n, 1)
out = np.append(i, p[:,1].reshape(n, 1), axis=1)

np.savetxt("outputC1.0.csv", out, fmt='%.7g', delimiter=',', header='Id,Prediction')

end = time.time()
print('\ntime elapsed: ', end - start)
