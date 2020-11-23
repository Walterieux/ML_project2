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

* `run.py` contains the creation, training and evaluation of our CNN. 
It can be run from your anaconda promt, with the correct environment selected.
 
