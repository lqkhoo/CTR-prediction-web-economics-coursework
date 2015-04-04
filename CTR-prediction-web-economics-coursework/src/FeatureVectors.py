import math
import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix, dok_matrix

class FV():
	# Sparse feature vector representation
	
	def __init__(self, fvs):
		self._set = fvs # the set this feature vector belongs to. FeatureVectors instance.
		self._dict = {}
	
	def __getitem__(self, key):
		if not self.is_sparse_key(key):
			return self._dict[key]
		elif self._set.keys()[key] == None:
			raise KeyError(key) # Raise if feature not expected to be sparse and has no default value
		return self._set.keys()[key]
	
	def set(self, key, value, default_value = None):
		self._dict[key] = value
		if key not in self._set.keys():
			self._set.add_key_value(key, default_value)
	
	def __len__(self):
		return self._set.keys().__len__()
	
	def __str__(self):
		# This displays the non-sparse fields only
		return self._dict.__str__()
	
	def is_sparse_key(self, key):
		if key in self._dict:
			return False
		else:
			return True
	
	def keys(self):
		return self._set.keys()
	
	def dense_keys(self):
		return self._dict
	
	def as_dict(self):
		# Returns full representation of vector as a dict
		
		d = self._set.keys().copy() # Acquire all default values
		for key in self._dict.keys():
			d[key] = self._dict[key] # Replace defaults with non-sparse entries
		return d

class FeatureVectors():
	# Sparse feature set representation
	
	def __init__(self, keys = {}, name = "Unnamed feature vector set"):
		# Optionally, pass in keys from another FeatureVectors instance. 
		self._name = name # Name of set. Not used for programmatic purposes
		self._keys = keys # Remembers valid keys. Maps keys to their default values
		self._vectors = []
	
	def __str__(self):
		return self._dict.__str__()
	
	def name(self):
		return self._name
	
	def keys(self):
		return self._keys
	
	def vectors(self):
		return self._vectors
	
	def add_key_value(self, key, value):
		self._keys[key] = value
	
	def set_keys(self, keys):
		# Replaces the current set of keys mapping default values with the given one.
		self._keys = keys
	
	@classmethod
	def mixin_keys(feature_vectors_instances):
		# Given an iterable of FeatureVectors instances, mixins all the keys for all of them
		all_keys = {}
		for feature_vectors in feature_vectors_instances:
			keys = feature_vectors.keys()
			for key in keys:
				if key not in all_keys:
					all_keys[key] = feature_vectors[key]
				elif all_keys[key] != feature_vectors[key]:
					raise ValueError(''.join(['Default values differ during key mixin! Key: ', key, ' values: ', str(all_keys[key]), ' ,', str(feature_vectors[key]), 'set: ', feature_vectors.name()]))
				# else do nothing
				
		for feature_vectors in feature_vectors_instances:
			feature_vectors.set_keys(all_keys)
		return feature_vectors_instances
	
	def as_scipy_sparse(self):
		
		# Copy all valid keys
		x_keys = self.keys().copy()
		del x_keys['label']
		
		# Sparse keys with nonzero default values (if any)
		x_nz = {}
		for key in x_keys.keys():
			if x_keys[key] == None or x_keys[key] == 0:
				continue
			x_nz[key] = x_keys[key]
		
		# x_keys now maps keys to indices in numpy array
		i = 0
		for key in x_keys:
			x_keys[key] = i
			i += 1
		
		n = len(self.vectors())
		dim = len(x_keys)
		xs = dok_matrix((n, dim), dtype=np.int32)
		#ys = dok_matrix((n, 1), dtype=np.int32)
		ys = []
		
		for i in range(n):
			vector = self.vectors()[i]
			
			# dense keys
			for key in vector.dense_keys():
				if key == 'label':
					continue
				j = x_keys[key]
				xs[i, j] = vector[key]
				
			# nonzero sparse keys
			for key in x_nz:
				j = x_keys[key]
				xs[i, j] = vector[key]
				
			#ys[i, 0] = vector['label']
			ys.append(vector['label'])
		
			if i % 10000 == 0:
				print("   Generating numpy sparse matrices. Vectors processed: " + str(i))
		return xs.tocsr(), np.array(ys)
	
	@classmethod
	def _set_vector(cls, vector, key, value, default_value, kf):
		# Helper for new_vector()
		# kf is either None or dict of filter keys. vector is FV instance
		# Sets new feature into vector and registers default_value as sparse feature
		#   if is in kf, or kf is None. Otherwise do nothing
		if kf == None or key in kf:
			vector.set(key, value, default_value)
		return vector
	
	def new_vector(self, dat, is_training_data = True, kf = None):
		# dat is one line in the tsv training file tokenized by splitting on \t
		# kf is the key filter. If given a dict, this limits the features to the given set.
		# Hence this method call will not identify / generate any new sparse features
		# Returns the FV instance, keyset
		
		if is_training_data == True:
			o = 0 # column offset
			label = int(dat[0])
		else:
			o = -1
			label = None
		
		# Register features ---
		vector = FV(self)
		
		# Non-sparse features
		FeatureVectors._set_vector(vector, 'label', label, None, kf)
		wknd = 1 if (dat[1+o] == '1' or dat[1+o] == '2') else 0 # 1==Sat, 2==Sun
		FeatureVectors._set_vector(vector, 'wknd', wknd, None, kf)
		FeatureVectors._set_vector(vector, 'hr', int(dat[2+o]), None, kf)
		w = int(dat[15+o])
		h = int(dat[16+o])
		FeatureVectors._set_vector(vector, 'ad-w', w, None, kf) # ad width
		FeatureVectors._set_vector(vector, 'ad-h', h, None, kf) # ad height
		FeatureVectors._set_vector(vector, 'ad-la', int(math.sqrt(w * h)), None, kf) # DEPENDENT VARIABLE # ad sqrt area
		FeatureVectors._set_vector(vector, 'ad-v', dat[17+o], None, kf) # ad visibility
		
		# Sparse features
		if label == 1 or label == None:
			FeatureVectors._set_vector(vector, ''.join(['d-', dat[11+o]]), 1, 0, kf) # domain
			FeatureVectors._set_vector(vector, ''.join(['dxsid-', dat[11+o], ' ', dat[14+o]]), 1, 0, kf) # domain x ad slot
			tags = dat[23+o].split(',')
			for tag in tags:
				FeatureVectors._set_vector(vector, ''.join(['t-', tag]), 1, 0, kf) # tag
				FeatureVectors._set_vector(vector, ''.join(['dxt-', dat[11+o], ' ', tag]), 1, 0, kf) # domain x tag
		
		# Register vector in set
		self.vectors().append(vector)
		return vector, self.keys()