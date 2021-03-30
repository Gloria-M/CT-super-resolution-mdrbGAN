"""
This module containg all the necessary methods for testing the GAN model.
"""
import os
import numpy as np
import torch

from models import Generator
from data_loader import make_loader
from utility_functions import *


class Tester:
    """
    Methods for training are defined in this class.

    Attributes:
        generator: Generator model
    """
    def __init__(self, args):

        self._test_dir = args.test_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.generator = Generator().to(self._device)

    def load_checkpoint(self, checkpoint_epoch):
        """
        Method to load pretrained model and optimizers parameters.
        :param checkpoint_epoch: the epoch to load information for
        """
        checkpoint_path = os.path.join(self._models_dir, 'checkpoint_{:d}.pt'.format(checkpoint_epoch))
        try:
            checkpoint_dict = torch.load(checkpoint_path)
            succes_message = 'Checkpoint(epoch {:d}) at {:s} loaded'.format(checkpoint_epoch, checkpoint_path)
            print(success_format(succes_message))

        except Exception as ex:
            fail_message = 'Checkpoint(epoch {:d}) at {:s} : {:s}'.format(checkpoint_epoch, checkpoint_path, ex.args[1])
            print(fail_format(fail_message))
            return

        # Load the generator pre-trained weights
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.to(self._device)

    def test(self, ct_name):
        """
        Method to test the Generator model on CT low-resolution image.
        :param ct_name: CT slice name to use for testing (without the .npy extension)
        :return: low-resolution, super-resolution and high-resolution reconstructed CT images
        """
        # Create data loader using the CT slice
        test_loader = make_loader(self._test_dir, [ct_name], mode='test')

        self.generator.eval()
        with torch.no_grad():
            for lr_ct, hr_ct in test_loader:

                # Eliminate extra dimension obtained when creating data loader
                # current size: [1 x batch_size x channels x height x width] -->
                # --> correct size: [batch_size x channels x height x width]
                lr_ct, hr_ct = lr_ct.squeeze(0), hr_ct.squeeze(0)
                lr_ct, hr_ct = lr_ct.to(self._device), hr_ct.to(self._device)

                # Generate super-resolution CT images
                sr_ct = self.generator(lr_ct)

                # Use the CT patches to recompose/compose the CT slice
                lr_slice = recompose_ct_slice(lr_ct)
                sr_slice = recompose_ct_slice(sr_ct)
                hr_slice = recompose_ct_slice(hr_ct)

                # Compute performance metrics
                scores_dict = compute_score(sr_slice, hr_slice)

        return lr_slice, sr_slice, hr_slice, scores_dict


def recompose_ct_slice(ct_crops):
    """
    Function to recompose the CT slice (512x512) from 16 patches (96x96).
    :param ct_crops: CT patches
    :return: recomposed CT slice
    """
    full_ct_size = 512
    crop_size = 128
    num_crops_h, num_crops_w = 4, 4

    ct_crops = ct_crops.detach().cpu().squeeze().numpy()
    full_ct = np.empty((full_ct_size, full_ct_size), dtype=ct_crops.dtype)

    for h_idx in range(num_crops_h):
        for w_idx in range(num_crops_w):
            start_h, end_h = h_idx * crop_size, (h_idx + 1) * crop_size
            start_w, end_w = w_idx * crop_size, (w_idx + 1) * crop_size

            crop_idx = h_idx * num_crops_h + w_idx
            full_ct[start_h:end_h, start_w:end_w] = ct_crops[crop_idx]

    return full_ct

def compute_score(sr_slice, hr_slice):
    """
    Function to compute the PSNR score to measure the quality of generated super-resolution image.
    :param sr_slice: super-resolution CT slice composed from patches
    :param hr_slice: high-resolution CT slice recomposed from patches
    :return: dictionary with performance metrics
    """
    hr_range = hr_slice.max() - hr_slice.min()
    mse = ((sr_slice - hr_slice) ** 2).mean()
    psnr = 10 * np.log10((hr_range ** 2) / mse)

    scores_dict = {'mse': mse,
                   'psnr': psnr}

    return scores_dict
