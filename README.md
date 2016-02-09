# brainseg

Created by Stavros Tsogkas at CentraleSupelec, Paris.

### Introduction

This code can be used to train and evaluate CNNs as described in our [paper](https://hal.archives-ouvertes.fr/hal-01265500v1) published at ISBI 2016. You can also find links to download the [pre-computed probability maps](download-probabilities) for all the volumes in the [IBSR dataset](https://www.nitrc.org/frs/?group_id=48) that we used in our experiments. 

### License

Our code is released under the MIT License (refer to the LICENSE file for details).

### Citing 

If you find our code or *CNN*-produced probability maps useful for your research, please cite:

    @inproceedings{shakeri2016subcortical,
        Author = {Shakeri, Mahsa and Tsogkas, Stavros and Lippe, Sarah and Kadoury, Samuel and Paragios, Nikos and Kokkinos, Iasonas},
        Title = {Sub-cortical Brain Structure Segmentation Using F-CNNs},
        Booktitle = {International Symposium on Biomedical Imaging ({ISBI})},
        Year = {2016}
    }
  
If you use the *RF*-produced probability maps please cite:

    @inproceedings{alchatzidis2014discrete,
        Author = {Discrete multi atlas segmentation using agreement constraints},
        Title = {Sub-cortical Brain Structure Segmentation Using F-CNNs},
        Booktitle = {British Machine Vision Conference ({BMVC})},
        Year = {2014}
    }

    
### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Download pre-computed probability maps for IBSR](#download)
5. [Usage](#usage)

### Requirements: software

1. A recent version of MATLAB. All our experiments were performed using MATLAB 2014a and Ubuntu 14.04.
2. [Our modified version of MatConvNet](https://github.com/tsogkas/matconvnet), with support for holes (included in this repository). 
3. [NifTI tools](http://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image) for loading NifTI files.

### Requirements: hardware

Our model is small enough to be trained using Nvidia GTX980 GPUs, demanding around 3GB of GPU RAM. If you want to experiment with different architectures, you may need GPUs with more RAM (e.g. Nvidia Titan or Tesla K40). You will probably need a computer with a good amount of CPU RAM, at least 12GB or more, depending on the amount of data augmentation. 

### Installation

1. Clone the brainseg repository.
  
2. Clone my [utils](https://github.com/tsogkas/utils) and [matconvnet](https://github.com/tsogkas/matconvnet) repos. Install MatConvNet following the steps described [here](http://www.vlfeat.org/matconvnet/install/).

3. Download the [IBSR dataset](https://www.nitrc.org/frs/?group_id=48). You will have to create a NITRC account, if you do not already have one.

4. Create a folder `brainseg/data` (this can be a symlink), place the downloaded file inside and extract it. This will result in a `brainseg/data/IBSR_nifti_stripped` directory. 

_Make sure that_ `utils`, `matconvnet`, _and_ `NifTI tools` _are included in the MATLAB working path._ 

### Download pre-computed probability maps for IBSR

Pre-computed probability maps for all the MRI volumes in IBSR can be found in the following links:
	
- [Probabilities computed with Random Forests](http://cvn.ecp.fr/data/brainsegm/RF_prob.zip)
- [Probabilities computed with CNNs](http://cvn.ecp.fr/data/brainsegm/cnn_prob_nii.zip)

The MR brain data sets and their manual segmentations were provided by the Center for Morphometric Analysis at Massachusetts General Hospital and are available at http://www.cma.mgh.harvard.edu/ibsr/.

### Usage

Coming soon.
