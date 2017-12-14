  <h2> Transfer Learning on UrbanSounds8K </h2>
  </p>  In this project, we use the pre-trained features from the Google AudioSet model available in https://github.com/tensorflow/models/tree/master/research/audioset and use it on the UrbanSounds8K dataset.</p>
 <br>
 
 <p>
 git clone https://www.github.com/abhyantrika/Transfer_Learning.git <br>
 cd Transfer_Learning 
  </p>
  
 Run <i> pip install -r requirements.txt </i> for installing the python dependencies.
 
 <p>
 Now download the pre trained weights and other requirements for the vggish model to run:<br>
 curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt <br>
 curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz <br>
 python vggish_smoke_test.py <br>
 This should not throw up any errors.
 </p>
  
 <p>
 Visit https://serv.cusp.nyu.edu/projects/urbansounddataset/ <br>
 and fill the form to download the UrbanSounds8K dataset and save it in the same directory and extract the tar file. <br>
 
  
<p>
After downloading,run <br>
mkdir <i> make_dirs.sh </i> <br>
to create new directories for storing processed data.<br>
</p>

<p>
Run <i> python pre_process.py </i> <br>
This will take the wav files of UrbanSounds8K dataset and feed forward it through the vggish pre trained network
and will save the 128X4 embeddings generated as an npz file in the features_npz directory.
</p>

<p>
Now run <i> python train_dnn.py </i> to train a simple 3 layer network for classifying the npz features.<br>
You can run <i> python test_dnn.py </i> for testing the model. This repo has trained weights and the model description in model.h5 and model.json files, respectively. </p>


