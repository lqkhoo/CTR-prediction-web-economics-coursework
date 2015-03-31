class FV():
	# Sparse feature vector representation
	
	def __init__(self, fvs):
		self._set = fvs # the set this feature vector belongs to. FeatureVectors instance.
		self._dict = {}
	
	def __getattr__(self, key):
		if key in self._dict:
			return self._dict[key]
		else:
			if self._set.keys[key] == None: raise KeyError(key) # Raise if feature not expected to be sparse and has no default value
			return self._set.keys[key]
	
	def set(self, key, value, default_value = None):
		self._dict[key] = value
		if key not in self._set.keys():
			self._set.add_key_value(key, default_value)
	
	def __len__(self):
		return len(self._set.keys())
	
	def __str__(self):
		# This displays the non-sparse fields only
		return str(self._dict)
	
	def keys(self):
		return self._set.keys()

class FeatureVectors():
	# Sparse feature set representation
	
	def __init__(self):
		self._keys = {} # Remembers valid keys. Maps keys to their default values
		self._vectors = []
	
	def __str__(self):
		return str(self._dict)
	
	def keys(self):
		return self._keys
	
	def add_key_value(self, key, value):
		self._keys[key] = value
	
	def vectors(self):
		return self._vectors
	
	def new_vector(self, dat, is_training_data = True):
		# dat is one line in the tsv training file tokenized by splitting on \t
		# input_line is one line in the .tsv training files
		
		"""
		def _register(vec, key, data_item, default_value = None):
			# Helper
			# vec is the vector being constructed
			# key is the FeatureVector key of the data item
			# default_value is the value given to the vector if this key is undefined
			vec.set(key, data_item, default_value)
		"""
		
		if is_training_data == True:
			o = 0 # column offset
			label = int(dat[0])
		else:
			o = -1
			label = None
		
		# Register features ---
		vector = FV(self)
		
		# Non-sparse features
		vector.set('label', label) # label
		wknd = 1 if (dat[1+o] == '1' or dat[1+o] == '2') else 0 # 1==Sat, 2==Sun
		vector.set('wknd', wknd) # is day weekend or weekday
		vector.set('hr', int(dat[2+o])) # hour of day
		w = int(dat[15+o])
		h = int(dat[16+o])
		vector.set('ad-w', w) # ad width
		vector.set('ad-h', h) # ad height
		vector.set('ad-a', w * h) # DEPENDENT VARIABLE # ad area
		vector.set('ad-v', dat[17+o]) # ad visibility
		
		# Sparse features
		if label == 1:
			vector.set(''.join(['d-', dat[11+o]]), 1, 0) # domain
			vector.set(''.join(['dxsid-', dat[11+o], ' ', dat[14+o]]), 1, 0) # domain x ad slot
			tags = dat[23+o].split(',')
			for tag in tags:
				vector.set(''.join(['t-', tag]), 1, 0) # tag
				vector.set(''.join(['dxt-', dat[11+o], ' ', tag]), 1, 0) # domain x tag
		
		# Register vector in set
		self._vectors.append(vector);
		return vector, self._keys