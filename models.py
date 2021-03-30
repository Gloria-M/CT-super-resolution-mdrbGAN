"""
This module defines the building of the Generative Adversarial Network based on
multiple dense residual blocks.
For the detailed description of the architecture please refer to https://gloria-m.github.io/super_resolution.html#s1.
"""
import numpy as np
import torch
from torch import autograd
import torchvision.models as models
import torch.nn as nn


class ResUnit(nn.Module):
    """
    This class defines the Residual Unit module.
    """
    def __init__(self, in_ch, hid_ch):
        super().__init__()

        self._seq1 = nn.Sequential(nn.Conv2d(in_ch, hid_ch, 3, padding=1),
                                   nn.BatchNorm2d(hid_ch),
                                   nn.PReLU())
        self._seq2 = nn.Sequential(nn.Conv2d(hid_ch, hid_ch, 3, padding=1),
                                   nn.BatchNorm2d(hid_ch))

    def forward(self, x):
        res = self._seq1(x)
        res = self._seq2(res)

        return res


class ResBlock(nn.Module):
    """
    This class defines the Residual Block module, consisting in 4 linked Residual Units.
    """
    def __init__(self, in_ch, hid_ch):
        super().__init__()

        bottle_out_ch = int(np.sqrt(in_ch))
        self._bottleneck = nn.Conv2d(in_ch, bottle_out_ch, 1)

        self._unit1 = ResUnit(bottle_out_ch, hid_ch)
        self._unit2 = ResUnit(hid_ch, hid_ch)
        self._unit3 = ResUnit(hid_ch, hid_ch)
        self._unit4 = ResUnit(hid_ch, hid_ch)

    def forward(self, x):

        in1 = x.clone()
        in1 = self._bottleneck(in1)
        res1 = self._unit1(in1)

        in2 = x + res1
        res2 = self._unit2(in2)

        in3 = in2 + res2
        res3 = self._unit3(in3)

        in4 = in3 + res3
        res4 = self._unit4(in4)

        out = in4 + res4

        return out


class Generator(nn.Module):
    """
    This class defines the Generator module of the GAN ensemble, composed of 4 densely connected Residual Blocks.
    The Generator recieves a low-resolution CT image and generates a super-resolution CT image.
    """
    def __init__(self, in_ch=1, hid_ch=64):
        super().__init__()

        self._block1 = ResBlock(in_ch, hid_ch)
        self._block2 = ResBlock(hid_ch, hid_ch)
        self._block3 = ResBlock(hid_ch, hid_ch)
        self._block4 = ResBlock(hid_ch, hid_ch)

        self._conv1 = nn.Sequential(nn.Conv2d(hid_ch, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    )
        self._conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 1, 3, padding=1),
                                    nn.PReLU()
                                    )

    def forward(self, x):
        in1 = x.clone()
        res1 = self._block1(in1)

        in2 = res1
        res2 = self._block2(in2)

        in3 = res2 + res1
        res3 = self._block3(in3)

        in4 = res3 + res2 + res1
        res4 = self._block4(in4)

        in5 = res4
        out1 = self._conv1(in5)

        in6 = out1 + x
        out2 = self._conv2(in6)

        return out2


class ConvBlock(nn.Module):
    """
    This class represents the Convolutional Block used in creating the Discriminator module.
    """
    def __init__(self, in_ch, out_ch, stride, with_bn=True):
        super().__init__()

        self.with_bn = with_bn

        self._conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self._bn = nn.BatchNorm2d(out_ch)
        self._act = nn.LeakyReLU(negative_slope=.2)

    def forward(self, x):

        x = self._conv(x)
        if self.with_bn:
            x = self._bn(x)
        x = self._act(x)

        return x


class Discriminator(nn.Module):
    """
    This class defines the Discriminator module of the GAN ensemble, composed of 8 Convolutional Blocks.
    """
    def __init__(self):
        super().__init__()

        self._features = nn.Sequential(ConvBlock(1, 64, 1, False),
                                       ConvBlock(64, 64, 2),
                                       ConvBlock(64, 128, 1),
                                       ConvBlock(128, 128, 2),
                                       ConvBlock(128, 256, 1),
                                       ConvBlock(256, 256, 2),
                                       ConvBlock(256, 512, 1),
                                       ConvBlock(512, 512, 2)
                                       )
        self._classifier = nn.Sequential(nn.Linear(6 * 6 * 512, 1024),
                                         nn.LeakyReLU(negative_slope=.2),
                                         nn.Linear(1024, 1)
                                         )

    def forward(self, x):
        x = self._features(x)

        b, c, h, w = x.shape
        x = x.view(b, c * h * w)

        x = self._classifier(x)

        return x


class FeatureExtractor(nn.Module):
    """
    This class defines the Feature Extractor module of the GAN ensemble.
    This is represented by a substructure of the VGG-19 model.
    """
    def __init__(self, device):
        super().__init__()

        self._device = device
        # Load the pre-trained VGG-19 weights
        vgg19_model = models.vgg19(pretrained=True)

        # The Feature Extractor uses the sequence of convolutions and subsampling layers defined in VGG-19.
        self._features = vgg19_model.features[:36]
        for name, param in self._features.named_parameters():
            param.requires_grad = False

    def forward(self, ct):
        # Normalize the CT images according to the VGG-19 authors' recomendation.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self._device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self._device)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)

        # The generated and high-resolution CT images have a single channel.
        # In order to be fed as input to the VGG-19 feature extractor, expand them to 3 channels.
        ct = ct.repeat(1, 3, 1, 1)
        ct = (ct - mean) / std

        ct_feats = self._features(ct)

        return ct_feats


class MDRB_GAN(nn.Module):
    """
    This class defines the GAN ensemble, composed of the previously defined Generator,
    Feature Extractor and Discriminator modules.
    """
    def __init__(self, device, alpha_param, lambda_coeff):
        super().__init__()

        # Parameters for computing the GAN loss
        self._alpha = alpha_param
        self._lambda_coeff = lambda_coeff

        self._device = device

        # Initialize GAN modules
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.feature_extractor = FeatureExtractor(self._device)

    def f_loss(self, lr_ct, hr_ct):
        """
        Method to compute F-loss from feature maps obtained by the Feature Extractor module,
         used to optimize the Generator.
        :param lr_ct: low-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: value of F-loss
        """

        # Generate super-resolution CT image from the low-resolution CT image
        sr_ct = self.generator(lr_ct)
        # Extract features from the super-resolution and high-resolution CT images
        sr_features = self.feature_extractor(sr_ct)
        hr_features = self.feature_extractor(hr_ct)

        # Compute loss
        criterion = nn.MSELoss()
        f_loss = criterion(sr_features, hr_features)

        return f_loss

    def g_loss(self, lr_ct, hr_ct):
        """
        Method to compute the Generator loss.
        :param lr_ct: low-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: value of Generator loss
        """

        # Generate super-resolution CT image from the low-resolution CT image
        sr_ct = self.generator(lr_ct)
        # Get the probability the super-resolution CT image to be the high-resolution CT image
        sr_critic = self.discriminator(sr_ct)

        # Compute the Generator loss by incorporating the previously defined F-loss
        f_loss = self.f_loss(lr_ct, hr_ct)
        g_loss = self._alpha * f_loss - (1 - self._alpha) * torch.mean(sr_critic)

        return f_loss, g_loss

    def generate_mixed_ct(self, sr_ct, hr_ct):
        """
        Method to create a mixed CT image from randomly weighted average between high-resolution and
        generated super-resolution CT images.
        :param sr_ct: generated super-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: mixed CT image
        """

        # Generate rondom weights
        epsilon_shape = [1 for _ in sr_ct.shape]
        epsilon = torch.rand(epsilon_shape, device=self._device).float()

        # Compute weighted average between super-resolution and high-resolution CT images
        ct_hat = epsilon * sr_ct + (1 - epsilon) * hr_ct
        ct_hat = ct_hat.float()
        ct_hat = ct_hat.to(self._device)

        return ct_hat

    def compute_gradient_penalty(self, lr_ct, hr_ct):
        """
        Method to compute the gradient penalty used in Wasserstein Loss computation,
        to achieve Lipschitz continuity.
        :param lr_ct: low-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: value to penalize the model with
        """

        # Generate super-resolution CT images from low-resolution CT images
        sr_ct = self.generator(lr_ct)
        # Create mixed CT image by weighted averaging the super-resolution and high-resolution images
        ct_hat = self.generate_mixed_ct(sr_ct, hr_ct)
        ct_hat_critic = self.discriminator(ct_hat)

        # Compute the gradients for the mixed CT image
        grad_outputs = torch.ones_like(ct_hat_critic).to(self._device)
        gradients = autograd.grad(outputs=ct_hat_critic, inputs=ct_hat,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute the gradient penalty
        gradient_penalty = self._lambda_coeff * torch.mean(
            (gradients.norm(2, dim=1) - 1) ** 2)

        return gradient_penalty

    def w_loss(self, lr_ct, hr_ct):
        """
        Method to define the Wasserstein Loss.
        :param lr_ct: low-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: value of Wasserstein loss
        """

        # Generate super-resolution CT image
        sr_ct = self.generator(lr_ct)
        # Discriminate super-resolution and high-resolution images
        sr_critic = self.discriminator(sr_ct)
        hr_critic = self.discriminator(hr_ct)

        # Compute the gradient penalty
        gp = self.compute_gradient_penalty(lr_ct, hr_ct)

        # Compute the Wasserstein loss
        # If the gradient norm moves away from the target value of 1, the model is penalized accordingly
        w_loss = -torch.mean(hr_critic) + torch.mean(sr_critic) + gp
        w_loss = (1 - self._alpha) * w_loss

        return w_loss

    def psnr(self, lr_ct, hr_ct):
        """
        Method to compute the Peak signal-to-noise ratio (PSNR), to measure the quality of a
        generated super-resolution CT image.
        :param lr_ct: low-resolution CT image
        :param hr_ct: high-resolution CT image
        :return: PSNR score, super-resolution CT image value range
        """

        # Generate super-resolution CT image from low-resolution CT image
        sr_ct = self.generator(lr_ct)
        criterion = nn.MSELoss()

        # Compute the MSE between the super-resolution and high-resolution images
        loss = criterion(sr_ct, hr_ct)
        loss = loss.data.cpu().numpy()
        # Compute PSNR score
        psnr = 10 * np.log10(1 / loss)
        # Get value range of super-resolution CT image
        sr_range = [torch.min(sr_ct).detach().cpu().numpy().item(),
                    torch.max(sr_ct).detach().cpu().numpy().item()]

        return psnr, sr_range
