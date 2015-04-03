import numpy as np
from sklearn import linear_model

from src.FeatureVectors import FeatureVectors

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

FILTER_CLICKED_DOMAINS = True

# Find which domains have at least one click in the training set
else:
	clicked_domains = None

# Generate training set, ignoring domains without any clicks at all

xs, ys = training_vectors.as_scipy_sparse()

# Generate test set, limiting features to those in training set
training_set_keys = training_vectors.keys()

test_vectors, n_test, dim_test = _generate_fvs(TEST_DATA_PATH, is_training_data = False, domains_filter = None, key_filter = training_set_keys)
