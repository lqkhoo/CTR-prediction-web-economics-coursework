import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import time
import pickle as pickle
import random as r

from FeatureVectors import FeatureVectors

TEST_DATA_PATH = "../data/data_test.tsv"
TRAIN_DATA_PATH = "../data/data_train.tsv"
DEBUG = 1

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


def _find_clicked_domains(file_path, limit = None):
	# Runs through given data set and returns all domains with at least one clicks
	# limit specifies a line limit to read in partial file
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
		if limit != None and i >= limit:
			break;
	
	la = len(all_domains)
	lc = len(clicked_domains)
	dprint(''.join(['Out of ', str(la), ' unique domains, ', str(lc), ' has >0 clicks. ', '(', str(100*lc/la) , '%)']))
	return clicked_domains, all_domains


def _generate_fvs(file_path, is_training_data, domains_filter = None, key_filter = None, limit = None):
	# Generate feature vectors with optional filter on domains
	# Otherwise generate vectors for every line
	# limit specifies a line limit to read in partial file
	# Returns fvs, n, dim
	dprint(''.join(['Generating fvs from ', file_path]))
	
	fvs = FeatureVectors()
	i = 0
	for line in open(file_path, 'r'):
		
		dat = _tokenize_line(line)
		domain = dat[11]
		
		if domains_filter == None or domain in domains_filter:
			# Vectorize
			v, k = fvs.new_vector(dat, is_training_data = is_training_data, kf = key_filter)
		
		i += 1
		if i % 100000 == 0: dprint(''.join(['   Processed: ', str(i), ' lines']))
		if limit != None and i >= limit:
			break;
	
	n = len(fvs.vectors())
	dim = len(fvs.keys())
	dprint(''.join([str(i), ' lines given. ', str(n), ' ', str(dim), '-d-vectors generated. (', str(100*n/i) ,'%)']))
	return fvs, n, dim


# Script ---

start = time.time()

FILTER_CLICKED_DOMAINS = True
SUBSAMPLE_MULTIPLE = 100 # Multiple of negative training examples to positive. Integer value only


# Find which domains have at least one click in the training set
if FILTER_CLICKED_DOMAINS == True:
	clicked_domains, all_domains = _find_clicked_domains(TRAIN_DATA_PATH)
else:
	clicked_domains = None


# Generate training set, ignoring domains without any clicks at all
training_vectors, n_train, dim_train = _generate_fvs(TRAIN_DATA_PATH, is_training_data = True, domains_filter = clicked_domains, limit = 20000)

# Deterministically select the first n negative training examples for reproducibility
dprint(''.join(['Subsampling negative training examples at ', str(SUBSAMPLE_MULTIPLE), 'x number of positive training examples.']))
training_pos_indices = training_vectors.filter_vectors_to_indices({'label': 1})
training_neg_indices = training_vectors.filter_vectors_to_indices({'label': 0})

# Shuffle the negative indices in a predictable way
r.seed(4332) 
training_neg_indices = sorted(training_neg_indices, key=lambda k: r.random())

training_neg_indices = training_neg_indices[0:SUBSAMPLE_MULTIPLE*len(training_pos_indices)]
training_subsampled_indices = []
training_subsampled_indices.extend(training_pos_indices)
training_subsampled_indices.extend(training_neg_indices)
training_vectors.set_vectors([training_vectors.vectors()[i] for i in training_subsampled_indices])

xs, ys, x_keys = training_vectors.as_scipy_sparse()

# Save train X
with open('gendata/xs_train20k.dat', 'wb') as outfile:
    pickle.dump(xs, outfile, pickle.HIGHEST_PROTOCOL)

# Save y
np.save('gendata/ys_train20k', ys)

# Save keys
with open('gendata/xkeys_train20k.dat', 'wb') as outfile:
    pickle.dump(x_keys, outfile, pickle.HIGHEST_PROTOCOL)

# Save train vectors
with open('gendata/train_vect20k.dat', 'wb') as outfile:
    pickle.dump(training_vectors, outfile, pickle.HIGHEST_PROTOCOL)


# Generate test set, limiting features to those in training set
training_set_keys = training_vectors.keys()
test_vectors, n_test, dim_test = _generate_fvs(TEST_DATA_PATH, is_training_data = False, domains_filter = None, key_filter = training_set_keys, limit = 20000)

# Ignore ys 
xs, ys, x_keys = test_vectors.as_scipy_sparse()

# Save test X
with open('gendata/xs_test20k.dat', 'wb') as outfile:
    pickle.dump(xs, outfile, pickle.HIGHEST_PROTOCOL)

# Save keys
with open('gendata/xkeys_test20k.dat', 'wb') as outfile:
    pickle.dump(x_keys, outfile, pickle.HIGHEST_PROTOCOL)

# Save test vectors
with open('gendata/test_vect20k.dat', 'wb') as outfile:
    pickle.dump(test_vectors, outfile, pickle.HIGHEST_PROTOCOL)


end = time.time()
print('\ntime elapsed: ', end - start)