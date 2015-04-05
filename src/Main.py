import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import time
import pickle as pickle
import csv

def _load_data():
	xs_train = []
	with open('gendata/xs_train20k.dat', 'rb') as infile:
	    xs_train = pickle.load(infile)
	ys_train = np.load('gendata/ys_train20k.npy')

	xs_test = []	
	with open('gendata/xs_test20k.dat', 'rb') as infile:
	    xs_test = pickle.load(infile)

	print("Loaded data files")
	return xs_train, ys_train, xs_test

def _load_keys_and_vect():	

	training_vectors = {}
	x_keys_train = {}
	with open('gendata/train_vect20k.dat', 'rb') as infile:
	    training_vectors = pickle.load(infile)
	with open('gendata/xkeys_train20k.dat', 'rb') as infile:
	    x_keys_train = pickle.load(infile)

	test_vectors = {}	
	x_keys_test = {}
	with open('gendata/test_vect20k.dat', 'rb') as infile:
	    test_vectors = pickle.load(infile)
	with open('gendata/xkeys_test20k.dat', 'rb') as infile:
	    x_keys_test = pickle.load(infile)

	print("Loaded key and vect files")
	return training_vectors, x_keys_train, test_vectors, x_keys_test


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


start = time.time()

np.set_printoptions(threshold=np.nan)

xs_train, ys_train, xs_test = _load_data()

# Initialize linear model
clf = linear_model.LogisticRegression(C=10.0, penalty='l1') #, tol=0.001)

best_clf = clf
best_mse = float("inf")

# Split input for 5 fold cross validation
cv = cross_validation.KFold(len(ys_train), 2, shuffle = True,  random_state=2876)

mses = []

for traincv, testcv in cv:
	X_train = xs_train[traincv]
	y_train = ys_train[traincv]

	X_test = xs_train[testcv]
	y_test = ys_train[testcv]

	clf.fit(X_train, y_train)

	if(best_mse == float("inf")):
		best_clf = clf

	p = clf.predict(X_test)

	mse = ((p - y_test) ** 2).mean()
	print("mse: ", mse)
	if(mse < best_mse):
		best_mse = mse
		best_clf = clf
	mses.append(mse)


clf = best_clf
print("Using clf with MSE: ", best_mse)

mses = np.array(mses)
print("\n\nAVG MSE: ", np.mean(mses))

p = clf.predict_proba(xs_test)
#print(clf.classes_)

n = len(p)

i = np.arange(1, n + 1, 1, dtype='int').reshape(n, 1)
out = np.append(i, p[:,1].reshape(n, 1), axis=1)

print("Generating out csv file")
np.savetxt("outs/out20k.csv", out, fmt='%.7G', delimiter=',', header='Id,Prediction')

print("Saving info files")
training_vectors, x_keys_train, test_vectors, x_keys_test = _load_keys_and_vect()

keyed_coeffs = {}
for key in x_keys_train:
	coef = clf.coef_[0][x_keys_test[key]]
	if coef != 0: keyed_coeffs[key] = coef
print(''.join(['Nonzero coefficients: ', str(keyed_coeffs)]))

_to_info_file('outs/test_out.csv', test_vectors, p, keyed_coeffs)

_to_info_file('outs/training_out.csv', training_vectors, clf.predict_proba(xs_train), keyed_coeffs)

end = time.time()
print('\ntime elapsed: ', end - start)
