# Road Segmentation

This project creates a road classifying model. The model classifies pixels of
a given satellite image with the labels `road=1, background=0`. The resulting
label array can be interpreted and displayed as an image. A CNN is used to 
train this model.

## Dependecies
In order to run the various python scripts of this project, we recommend the
usage of a separate anconda environment. The following lines, excuted in order
in your anaconda prompt, will take care of this:

* Create a new anaconda environment : `conda create -n tf tensorflow`
* Activate (switch to) the new environment : `conda activate tf`

The above steps will create a new environment with TensorFlow 2.1 installed.
You are free to choose Python 3.6, 3.7 or 3.8 for this envionment. Any other 
Python versions might result in inability to execute the project files. If you
wish to install the GPU version of tensorflow for improved training speed, 
use the following lines instead:

* Create a new anaconda environment : `conda create -n tf-gpu tensorflow-gpu`
* Activate (switch to) the new environment : `conda activate tf-gpu`

Note that TF for GPU requires:

* A compatible NVidia GPU
* NVidia CUDA Development Kit v10.1
* NVidia cuDNN 7 

The following dependencies need to be installed before you can run
the project. Next to them are the anaconda prompt commands needed to 
install them on your anaconda environment: 

* ipykernel : `conda install -c anaconda ipykernel`
* patchify : `pip install patchify`
* imageio : `conda install -c anaconda imageio`
* glob : `pip install glob`
* pillow : `conda install -c anaconda pillow`

Note that pip (or pip3) is required to install some dependecies.

## Files

* folder `scripts` contains pyhton scripts that contain various methods and the 
implementations thereof, as mentioned in the report. The comments in the scripts
themselves detail their function.
* `scripts/CNN_Evolution` contains 3 implementations that demo some major aspects of our development
process. The details can be found in the scripts themselves, as comments.
* `scripts/Pre_Processing` contains 1 implementation that shows the use of what we call 
edges and distance in our report, `cnn_on_2_extracted_features.py`. It also contains 2 other scripts. 
`images_preprocess` contains the code used to enlarge our data set, as described in the report.
`patches.py` is used to create patches from images and vice versa.
* `scripts/Post_Processing` conatins one script that regroups utility functions, `utility.py`,
two scripts useful for creating submission, `create_submission_groundtruth.py` and `mask_to_submission.py`. 
It also contains the script used to post-process images,as described in the report, `convolution.py`.
* `scripts/final_model/U_Net_patches_256.py` contains the creation, training and 
evaluation of our final CNN. It can be run from your anaconda promt, with the correct 
environment selected, provided you have a powerful GPU, enough memory, and time to spare.
We trained this model using Google's Colab.
* `model_plot.png` can be found in `scripts/final_model/`.
* the scripts are meant to be run using the correct data, placed in a folder named `data`, placed next to `scripts`.

## Our Training Environment

All results shown in our report and in the images that can be found in `scripts` are obtained after training our
models using Google Colab. All training sessions are run for 50 epochs, since they tend to have converged after that.
The models are trained on an NVidia Tesla V100-SXM2-16GB GPU. Training times vary depending on the model. Our final model's
training time was about 5 hours in the described environment.
 
