import glob
import os
import numpy as np 
import vggish_inference_demo

#labels = np.array([total_number_of_samples])


def generate_features(parent_dir):

	sub_dirs = os.listdir(parent_dir)
	success = 0
	failures = 0

	for folder in sub_dirs:
		files = os.listdir(parent_dir + folder)
		for f in files:
			try:
				label = f.split('-')[1]
				print(parent_dir+folder+'/'+	f)
				outcome = vggish_inference_demo.main(parent_dir+folder+'/'+f,'features_npz/c'+str(label)+'/'+f+'.npz')
				if outcome==1:
					success+=1
				else:
					failures+=1	

			except:
				failures+=1
				pass	

	return success,failures		

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels,n_unique_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode

success,failures = generate_features('UrbanSound8K/audio/')
print(success,failures)

