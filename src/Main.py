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


def _find_clicked_domains():
	# First pass - find which domains have more than one click
	# Returns clicked_domains, all_domains
	print('First pass. Find clicked domains')
	
	all_domains = {}
	clicked_domains = {}
	i = 0
	for line in open(TRAIN_DATA_PATH, 'r'):
		
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
	
	la = len(all_domains)
	lc = len(clicked_domains)
	dprint(''.join(['Out of ', str(la), ' unique domains, ', str(lc), ' has >0 clicks. ', '(', str(100*lc/la) , '%)']))
	return clicked_domains, all_domains


def _generate_fvs(domains_filter = None):
	# Second pass - generate vectors but with optional filter on domains
	# Otherwise generate vectors for every line
	# Returns fvs, n, dim
	dprint('Second pass. Generate fvs')
	
	fvs = FeatureVectors()
	i = 0
	for line in open(TRAIN_DATA_PATH, 'r'):
		
		dat = _tokenize_line(line)
		domain = dat[11]
		
		if domain in domains_filter or domains_filter == None:
			# Vectorize
			v, k = fvs.new_vector(dat, is_training_data = True)
		
		i += 1
		if i % 100000 == 0: dprint(''.join(['   Processed: ', str(i), ' lines']))
	
	n = len(fvs.vectors())
	dim = len(fvs.keys())
	dprint(''.join([str(i), ' lines given. ', str(n), ' ', str(dim), '-d-vectors generated. (', str(100*n/i) ,'%)']))
	return fvs, n, dim

# Filter only domains with >0 clicks and generate their fvs
clicked_domains, all_domains = _find_clicked_domains()
training_vectors = _generate_fvs(clicked_domains)


print(len(training_vectors.keys()))
print(len(training_vectors.vectors()))