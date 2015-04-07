import numpy as np
import math
from sklearn import linear_model
from sklearn import cross_validation
import time
import pickle as pickle
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.cross_validation import train_test_split

DEBUG = 1


def _load_data():
	xs_train = []
	with open('gendata/xs_trainsubKF.dat', 'rb') as infile:
	    xs_train = pickle.load(infile)
	ys_train = np.load('gendata/ys_trainsubKF.npy')

	xs_test = []	
	with open('gendata/xs_testsubKF.dat', 'rb') as infile:
	    xs_test = pickle.load(infile)

	print("Loaded data files")
	return xs_train, ys_train, xs_test

def _load_keys():	

	x_keys_train = {}
	with open('gendata/xkeys_trainsubKF.dat', 'rb') as infile:
	    x_keys_train = pickle.load(infile)

	x_keys_test = {}
	with open('gendata/xkeys_testsubKF.dat', 'rb') as infile:
	    x_keys_test = pickle.load(infile)

	print("Loaded key and vect files")
	return x_keys_train, x_keys_test

def _load_vect():	

	training_vectors = {}
	with open('gendata/train_vectsubKF.dat', 'rb') as infile:
	    training_vectors = pickle.load(infile)

	test_vectors = {}	
	with open('gendata/test_vectsubKF.dat', 'rb') as infile:
	    test_vectors = pickle.load(infile)

	print("Loaded key and vect files")
	return training_vectors, test_vectors


def _to_info_file(file_path, fvs, ys_pred, keyed_coeffs):
	print(''.join(['Writing f(x) of ', fvs.name(), ' to file...']))
	
	keys = sorted(keyed_coeffs.keys())
	
	f = open(file_path, 'w')
	coeffs = ','
	for key in keys:
		coeffs += ',' + str(keyed_coeffs[key])
	coeffs += '\n'
	f.write(coeffs)
	
	key_fields = 'Id' + ',' + 'Prediction'
	for key in keys:
		key_fields += ',' + key
	key_fields += '\n'
	f.write(key_fields)
	
	for i in range(len(fvs.vectors())):
		key_str = ''
		vector = fvs.vectors()[i]
		for key in keys:
			key_str += ',' + str(vector[key])
		key_str += '\n'
		f.write(''.join([str(i+1), ',', str(ys_pred[i]), key_str]))
	f.close()

def key_histogram(dict):
	count = {}
	for k in dict.keys():
		#if len(k) > 5:
		key = k.split('-')[0]
		if key == None:
			key = k
		if key in count:
			count[key] += 1
		else:
			count[key] = 1

	print(count)


def dprint(string):
	if DEBUG == 1: print(string)


start = time.time()

np.set_printoptions(threshold=np.nan)

xs_train, ys_train, xs_test = _load_data()

'''
N = 2847802 # Training set size
len ys:  209676
1s len:  2076
0s len:  207600

N zeros = 2845726

x : 100 = 207600 : 2845726
x = 7.29

weight = 1/(207600/2845726)
'''

# Initialize linear model
C = 0.09
dprint("C = ", C)
weight = {0: 1/(207600/2845726), 1: 1}

clf = linear_model.LogisticRegression(C=C, penalty='l1', class_weight=weight, random_state=1)

#clf = linear_model.RandomizedLogisticRegression(C=C, normalize=True, n_resampling=int(l/4), random_state=1)
X_train, X_test, y_train, y_test = train_test_split(xs_train, ys_train, test_size = 0.25, random_state = 0)


clf.fit(X_train, y_train)
p = clf.predict_proba(X_test)

mse = ((p[:,1] - y_test) ** 2).mean()
rmse = math.sqrt(mse)
dprint("rmse: ", rmse)

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, p[:,1])
auc_res = auc(fpr, tpr)
dprint("auc = ", auc_res)

auc_bas = 0.999
# (AUC - 0.5) / (AUC_BASE - 0.5)
r = (auc_res - 0.5) / (auc_bas - 0.5)

dprint("result: ", r)

#plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_bas))
#plt.show()


clf.fit(xs_train, ys_train)
coefs = clf.coef_

p = clf.predict_proba(xs_test)
n = len(p)

i = np.arange(1, n + 1, 1, dtype='int').reshape(n, 1)
out = np.append(i, p[:,1].reshape(n, 1), axis=1)

dprint("Generating out csv file")
np.savetxt("outs/outl1BIG09.csv", out, fmt='%.7G', delimiter=',', header='Id,Prediction')

dprint("Generating key info")
x_keys_train, x_keys_test = _load_keys()

keyed_coeffs = {}
for key in x_keys_train:
	coef = coefs[0][x_keys_test[key]]
	if coef != 0: keyed_coeffs[key] = coef
dprint(''.join(['Nonzero coefficients: ', str(keyed_coeffs)]))

key_histogram(keyed_coeffs)

dprint("from ", len(x_keys_train.values()), " keys, the nonzero ones are ", len(keyed_coeffs.values()))
dprint("max coef = ", np.max(coefs), ", min coef = ", np.min(coefs))
dprint("max pred = ", np.max(p), ", min pred = ", np.min(p))


end = time.time()
dprint('\ntime elapsed: ', end - start)



