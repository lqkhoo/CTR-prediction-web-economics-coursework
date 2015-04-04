from random import shuffle
import numpy as np
import math
from sklearn import linear_model

from src.FeatureVectors import FeatureVectors

# Constants
# N = 2847802 # Training set size
# UDOM = 15146 # Number of unique domains
# CLICKS = 2076 # Number of clicks
# UDOMC = 432 # Number of unique domains with clicks

def dprint(string):
	if DEBUG == 1: print(string)

def _tokenize_line(line):
	dat = line.strip().split('\t')
	return dat


def _find_clicked_domains(file_path, read_limit = None):
	# Runs through given data set and returns all domains with at least one clicks
	# read_limit specifies a line read_limit to read in partial file
	# Returns clicked_domains, all_domains
	dprint(''.join(['Finding clicked domains in ', file_path]))
	
	all_domains = {}
	clicked_domains = {}
	i = 0
	for line in open(file_path, 'r'):
		
		dat = _tokenize_line(line)
		label = int(dat[0]) # training set
		domain = dat[11]
		
		if domain not in all_domains: all_domains[domain] = 1
		else: all_domains[domain] += 1
		if label == 1:
			if domain not in clicked_domains: clicked_domains[domain] = 1
			else: clicked_domains[domain] += 1
		
		i += 1
		if i % 100000 == 0: dprint(''.join(['   Processed: ', str(i), ' lines']))
		if read_limit != None and i >= read_limit:
			break
	
	la = len(all_domains)
	lc = len(clicked_domains)
	dprint(''.join(['   Out of ', str(la), ' unique domains, ', str(lc), ' has >0 clicks. ', '(', str(100*lc/la) , '%)']))
	return clicked_domains, all_domains


def _generate_fvs(file_path, is_training_data, name = FeatureVectors.DEFAULT_NAME, domains_filter = None, key_filter = None, read_limit = None):
	# Generate feature vectors with optional filter on domains
	# Otherwise generate vectors for every line
	# read_limit specifies a limit in the number of lines to read in partial file
	# Returns fvs, n, dim
	dprint(''.join(['Generating fvs from ', file_path]))
	
	fvs = FeatureVectors(name = name)
	i = 0
	for line in open(file_path, 'r'):
		
		dat = _tokenize_line(line)
		domain = dat[11]
		
		if domains_filter == None or domain in domains_filter:
			# Vectorize
			v, k = fvs.new_vector(dat, is_training_data = is_training_data, kf = key_filter)
		
		i += 1
		if i % 100000 == 0: dprint(''.join(['   Processed: ', str(i), ' lines']))
		if read_limit != None and i >= read_limit:
			break
	
	n = len(fvs.vectors())
	dim = len(fvs.keys()) # this includes label
	dprint(''.join(['   ', str(i), ' lines given. ', str(n), ' ', str(dim), '-d-vectors generated. (', str(100*n/i) ,'%)']))
	return fvs, n, dim

def _get_rmse(model, xs, ys):
	y_pred = _get_decision_function(model, xs)
	
	if y_pred.shape[0] != ys.shape[0]: raise ValueError("Shape mismatch")
	
	sse = 0
	for i in ys:
		e = ys[i] - y_pred[i]
		sse += e * e
	rmse = math.sqrt(sse / ys.shape[0])
	return rmse

def _get_decision_function(model, xs):
	# Gets RMSE of model on given set of xs and ys. xs and ys should be numpy array-like objects
	y_pred = model.decision_function(xs)
	# y_pred = train_xs.dot(model.coef_[0].transpose(True)) + model.intercept_ # equivalent
	y_pred = 1.0/(1+np.exp(-y_pred))
	return y_pred

# Script ---

DEBUG = 1
READ_LIMIT = None # Set to integer to limit file line reads

# Paths
TEST_DATA_PATH = "../data/data_test.tsv"
TRAIN_DATA_PATH = "../data/data_train.tsv"
FILE_DIR_PATH = "../files/"

# Algo params
FILTER_CLICKED_DOMAINS = True 				# Filters training examples to domains with at least one positive training example
SUBSAMPLE_NEGATIVE_TRAINING_EXAMPLES = True # Subsample negative training examples to balance training set
SUBSAMPLE_MULTIPLE = 10 					# Multiple of negative training examples to positive. Integer value only

# TRAIN ----

# Find which domains have at least one click in the training set
if FILTER_CLICKED_DOMAINS == True:
	clicked_domains, all_domains = _find_clicked_domains(TRAIN_DATA_PATH, read_limit = READ_LIMIT)
else:
	clicked_domains = None

# Generate training set, ignoring domains without any clicks at all
training_vectors, n_train, dim_train = _generate_fvs(TRAIN_DATA_PATH, name = 'training set', is_training_data = True, domains_filter = clicked_domains, read_limit = READ_LIMIT)

# Deterministically select the first n negative training examples for reproducibility
dprint(''.join(['Subsampling negative training examples at ', str(SUBSAMPLE_MULTIPLE), 'x number of positive training examples.']))
training_pos_indices = training_vectors.filter_vectors_to_indices({'label': 1})
training_neg_indices = training_vectors.filter_vectors_to_indices({'label': 0})
#shuffle(training_neg_indices)

training_neg_indices = training_neg_indices[0:SUBSAMPLE_MULTIPLE*len(training_pos_indices)]
training_subsampled_indices = []
training_subsampled_indices.extend(training_pos_indices)
training_subsampled_indices.extend(training_neg_indices)
training_vectors.set_vectors([training_vectors.vectors()[i] for i in training_subsampled_indices])

model = linear_model.LogisticRegression(C=1.0, penalty='l1', class_weight={1:1, 0: 1.0 / SUBSAMPLE_MULTIPLE})
# model = linear_model.LogisticRegression(C=1.0, penalty='l1', class_weight={1:1, 0: len(training_pos_indices) / len(training_neg_indices)})
train_xs, train_ys, key_to_index_map = training_vectors.as_scipy_sparse()
model.fit(train_xs, train_ys)

rmse = _get_rmse(model, train_xs, train_ys)
print(''.join(['Training RMSE: ', str(rmse)]))
keyed_coeffs = {}
for key in key_to_index_map:
	coef = model.coef_[0][key_to_index_map[key]]
	if coef != 0: keyed_coeffs[key] = coef
print(''.join(['Nonzero coefficients: ', str(keyed_coeffs)]))

# TEST ----

# Generate test set, read_limiting features to those in training set
training_set_keys = training_vectors.keys()
test_vectors, n_test, dim_test = _generate_fvs(TEST_DATA_PATH, name = 'test set', is_training_data = False, domains_filter = None, key_filter = training_set_keys, read_limit = READ_LIMIT)
test_xs, test_ys, key_to_index_map = test_vectors.as_scipy_sparse()
test_ys_pred = _get_decision_function(model, test_xs)

f = open(FILE_DIR_PATH + 'out.csv', 'w')
f.write(''.join(['Id', ',', 'Prediction', '\n']))
for i in range(len(test_vectors.vectors())):
	f.write(''.join([str(i+1), ',', str(test_ys_pred[i]), '\n']))
f.close()