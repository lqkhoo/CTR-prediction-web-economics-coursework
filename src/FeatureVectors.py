class FeatureVectors():
	# Sparse feature set representation
	
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
		
		def __len__(self):
			return self._set.__len__()
		
		def keys(self):
			return self._set.keys()
		
		def get(self, key):
			return self.__getattr__(key)
	
	
	def __init__(self):
		self._keys = {} # Remembers valid keys. Maps keys to their default values
		self._vectors = []
	
	def __str__(self):
		return str(self._dict)
	
	def keys(self):
		return self._keys
	
	def vectors(self):
		return self._vectors
	
	
	@classmethod
	def stringify(entry):
		pass
	
	def new_vector(self, dat, is_training_data = True, update_keys = True):
		# dat is one line in the tsv training file tokenized by splitting on \t
		# input_line is one line in the .tsv training files
		# Updates keys by default on each call. Disable for performance
		
		def _register(vec, key, data_item, default_value = None):
			# Helper
			# vec is the vector being constructed
			# key is the FeatureVector key of the data item
			# default_value is the value given to the vector if this key is undefined
			vec[key] = data_item
			if update_keys == True:
				self._keys[key] = default_value
		
		
		if is_training_data == True:
			o = 0 # column offset
			label = int(dat[0])
		else:
			o = -1
			label = None
		
		# Register features ---
		vector = {}
		
		# Non-sparse features
		_register(vector, 'label', label) # label
		wknd = 1 if (dat[1+o] == '1' or dat[1+o] == '2') else 0 # 1==Sat, 2==Sun
		_register(vector, 'wknd', wknd) # is day weekend or weekday
		_register(vector, 'hr', int(dat[2+o])) # hour of day
		w = int(dat[15+o])
		h = int(dat[16+o])
		_register(vector, 'ad-w', w) # ad width
		_register(vector, 'ad-h', h) # ad height
		_register(vector, 'ad-a', w * h) # DEPENDENT VARIABLE # ad area
		_register(vector, 'ad-v', dat[17+o]) # ad visibility
		
		# Sparse features
		if label == 1:
			_register(vector, ''.join(['d-', dat[11+o]]), 1, 0) # domain
			_register(vector, ''.join(['dxsid-', dat[11+o], ' ', dat[14+o]]), 1, 0) # domain x ad slot
			tags = dat[23+o].split(',')
			for tag in tags:
				_register(vector, ''.join(['t-', tag]), 1, 0) # tag
				_register(vector, ''.join(['dxt-', dat[11+o], ' ', tag]), 1, 0) # domain x tag
		
		# Register vector in set
		self._vectors.append(vector);
		return vector, self._keys