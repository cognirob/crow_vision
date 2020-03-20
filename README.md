# crow_vision

This repo contains all computer vision code for the CROW project. 

## Objective

Object recognition (on RGB camera, real-time), segmentation (bounding boxes/bboxes & pixel masks) -> conversion to 3D point-clouds/bboxes. 

## Installation

### YOLACT 

#### Prerequisities

- nvidia-driver(-418) (`apt install nvidia-driver-418`)
- [CUDA 10](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)

#### Install YOLACT

##### Submodule

This repo uses git submodules for external git projects. 
https://git-scm.com/book/en/v2/Git-Tools-Submodules 

For the first time using this repo, initialize the submodules: 
```
cd external/yolact
git submodule update --init --recursive
```
This will populate the `yolact/` folder with the upstream code. 

To update, 
`git pull --recurse-submodules`

##### Install env

Tested on:
- Ubuntu linux 18.04
- obtain [Anaconda python](https://www.anaconda.com/distribution/)
- install preset environment: `conda env create -f environment.yml && conda activate yolact-env`
- you can use YOLACT(++) following its [README](external/yolact/README.md)
- To use our vision with other tools/packages, install : `python setup.py develop`

#### Usage

See [YOLACT's original README](external/yolact/README.md)

##### Run on a single image
`cd external/yolact`
`python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.jpeg`

##### Obtain our custom model (weights)

use DVC to pull the data & `config.py` file (how? TODO)
`python eval.py --trained_model=weights/<custom weights>.pth --score_threshold=0.15 --top_k=15 --config=<custom config matching the new weights> --image=my_image.jpeg`

### Intel RealSense camera

TODO




## DVC HowTo

### Setup DVC for large file (neural nets) storage

If there is not .dvc folder then initialize DVC similar to Git


`dvc init`


First you need to create autologin to the cluster

Check whether you have public key at your ~/.ssh/ folder. If there is not id_rsa.pub then create one

`ssh-keygen -t rsa`

Copy this key to the cluster for autologin:

`cat .ssh/id_rsa.pub | ssh yourname@cluster.ciirc.cvut.cz 'cat >> .ssh/authorized_keys`

Check whether autologin works

`ssh yourname@cluster.ciirc.cvut.cz`

Set up remote storage for the repository to cluster.ciirc.cvut.cz

`dvc remote add -d cluster ssh://cluster.ciirc.cvut.cz:/nfs/projects/crow/dvc`

`dvc remote modify cluster user yourname`

`dvc remote modify cluster keyfile ~/.ssh/id_rsa.pub`

Adding files or directories to DVC repository

`dvc add data/yourresults`

Pair your actual data (DVC) with the actual code (GIT)

`git add data/.gitignore data/yourresults.dvc`

Name changes before pushing them to DVC and GIT

`git commit -m "Trained network #7"`

`git tag Network7`

Now you can push both data and code to the repository

`git push --tags`

`dvc push`

If you want to checkout to this commit just type:

`git checkout Network7`

`dvc pull`

It means that every time you train the network and store the trained files in the data folder, you first need to create small dvc file, then pair this file with the git commit and tag this commit to be able to checkout to this trained network. Then you push the code to git and data to dvc separately.

### Working with DVC 

After traininng the network and saving the weights to the data/youresults folder type:

`dvc add data/yourresults`

`git add data/yourresults.dvc`

`git commit -m "Michal DOPE net #122 - 80% accuracy"`

`git tag MichalDope122`

`git push --tags`

`dvc push`

If you want to test your network or somebody else will send you tag of trained network just type:

`git checkout MegiYolact23`

`dvc pull`

