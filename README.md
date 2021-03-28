# CT super-resolution using multiple dense resudual blocks based GAN

This repository contains the PyTorch implementation of CT super-resolution using multiple dense residual block based GAN presented in [[Zhang et al., 2020]](https://link.springer.com/article/10.1007/s11760-020-01790-5), with slight changes as recommended in [[Gulrajani et al., 2017]](https://arxiv.org/abs/1704.00028) and with architecture parameters adapted from [[Ledig et al., 2017]](https://arxiv.org/abs/1609.04802), in case they were not specified.

### For the complete description of the implementation methods and experiments please refer to [CT_super-resolution](https://gloria-m.github.io/super_resolution.html).  

<br/>  

## Dataset

The data used represent a subset of [MosMed Covid-19 dataset](https://journals.eco-vector.com/DD/article/view/46826). It consist in multiple chest CT scans of healthy and infected patients in `.nii.gz` 3D-format, with different number of scan slices and slices thickness.  
The 3-dimensional CTs are splitted in slices and each slice is saved in `.npy` 2D-format.

### Data path structure

The data directory should have the following structure:
```
.
├── Data
    ├── Test
    │   ├── *.npy
    ├── Train
    │   ├── *.npy
    ├── Validation
    │   ├── *.npy
```  

## Usage

### 1. Train

#### run `python main.py`  

This will automatically start a Tensorboard session, with the following parameters:  
```
--tb_port (specify the port number to be used by Tensorboard) 
--tb_logdir = 'runs' (directory to write logs to Tensorboard)  
--tb_plot_interval = 10 (interval for creating and writing generated CT images | the losses and PSNR scores are logged every <--log_interval> epochs)
```  

Control the training by modifying the default values for the following parameters:
```
--device = cuda (train on cuda)  
--log_interval = 1 (print train & validation loss each epoch)
--checkpoint_interval = 100 (save trained model and optimizer parameters every 100 epochs)
--num_epochs = 500
```

### 2. Resume training

#### run `python main.py --resume_training=true --restore_epoch=*`  
Resume training by specifying a valid value for `--restore_epoch`.  
> The model saved as `Models/checkpoint_<restore_epoch>.pt` will be loaded

### 3. Test

#### run `python main.py --mode=test --restore_epoch=* --test_ct_names=*`  
Test the model saved at training epoch `--restore_epoch` on CT images specified.
> `--test_ct_names` accepts a list of the CT images without the `.npy` extension.
> > for example, the CT image located at `Data/Test/ct_sample1.npy` will be passed as `ct_sample1`.  

> The model saved as `Models/checkpoint_<restore_epoch>.pt` will be loaded.

<br/>  

## To Do

> Add pre-processing code for CT scans __*.nii.gz  --> *.npy__  
> Write `help` for arguments.  
