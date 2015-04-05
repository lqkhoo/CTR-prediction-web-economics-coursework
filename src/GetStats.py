from src.FeatureVectors import FeatureVectors
from pprint import pprint

TEST_DATA_PATH = "../data/data_test.tsv"
TRAIN_DATA_PATH = "../data/data_train.tsv"

# Constants
N = 2847802 # Training set size


# Generate vectors
training_vectors = FeatureVectors()
i = 0

clicks = 0
dict = {}
for line in open(TRAIN_DATA_PATH, 'r'):
	v, k = training_vectors.new_vector(line, is_training_data = True)
	
	if v['label'] == 1:
		clicks +=1
		if v['dom'] not in dict:
			dict[v['dom']] = 1
		else:
			dict[v['dom']] = dict[v['dom']] + 1
	
	i += 1
	if i % 100000 == 0:
		print(''.join('processed: ' + str(i)))

print(''.join(['n:', str(i)]))
print(''.join(['clicks: ' + str(clicks)]))
print(''.join(['unique domains: ' + str(len(dict.keys()))]))
pprint(dict)