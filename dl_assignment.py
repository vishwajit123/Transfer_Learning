import glob
import os
import numpy as np 
import vggish_inference_demo

#labels = np.array([total_number_of_samples])


def generate_features(parent_dir):

	"""
	for label, sub_dir in enumerate(sub_dirs):
		for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
				print os.system('pwd')

			#here you'll extract features and store (using VGG)
			#pass the path to vggifn_inference_demo function
			#and store them as a tuple with the labels
			#

			#labels = np.append(labels, fn.split('/')[2].split('-')[1])
	"""
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
#(5328, 3410)
