# CT super-resolution using multiple dense resudual blocks based GAN

This repository contains the implementation of CT super-resolution using multiple dense residual block based GAN presented in [[Zhang et al., 2020]](https://link.springer.com/article/10.1007/s11760-020-01790-5), with slight changes as recommended in [[Gulrajani et al., 2017]](https://arxiv.org/abs/1704.00028) and with architecture parameters adapted from [[Ledig et al., 2017]](https://arxiv.org/abs/1609.04802), in case they were not specified.

For the complete description of the implementation methods and experiments please refer to [CT_super-resolution](https://gloria-m.github.io/super_resolution.html).



## Usage

### 1. Train

run `python main.py`  
<br/>  
control the training by modifying the default values for the following parameters:
```
--device = cuda (train on cuda)  
--log_interval = 1 (print train & validation loss each epoch)
--checkpoint_interval = 100 (save trained model and optimizer parameters every 100 epochs)
--num_epochs = 500
```

### 2. Resume training

run `python main.py --resume_training=true --restore_epoch=*`  
<br/>  
Resume training by specifying a valid `--restore_epoch`

