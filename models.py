import numpy as np
import torch
from torch import autograd
import torchvision.models as models
import torch.nn as nn


class ResUnit(nn.Module):
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
    def __init__(self, device):
        super().__init__()

        self._device = device
        vgg19_model = models.vgg19(pretrained=True)

        self._features = vgg19_model.features[:36]
        for name, param in self._features.named_parameters():
            param.requires_grad = False

    def forward(self, ct):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self._device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self._device)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)

        ct = ct.repeat(1, 3, 1, 1)
        ct = (ct - mean) / std

        ct_feats = self._features(ct)

        return ct_feats


class MDRB_GAN(nn.Module):
    def __init__(self, device, alpha_param, lambda_coeff):
        super().__init__()

        self._alpha = alpha_param
        self._lambda_coeff = lambda_coeff
        self._device = device

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.feature_extractor = FeatureExtractor(self._device)

    def f_loss(self, lr_ct, hr_ct):
        sr_ct = self.generator(lr_ct)
        sr_features = self.feature_extractor(sr_ct)
        hr_features = self.feature_extractor(hr_ct)

        criterion = nn.MSELoss()
        f_loss = criterion(sr_features, hr_features)

        return f_loss

    def g_loss(self, lr_ct, hr_ct):
        sr_ct = self.generator(lr_ct)
        sr_critic = self.discriminator(sr_ct)

        f_loss = self.f_loss(lr_ct, hr_ct)
        g_loss = self._alpha * f_loss - (1 - self._alpha) * torch.mean(sr_critic)

        return f_loss, g_loss

    def generate_mixed_ct(self, lr_ct, hr_ct):
        epsilon_shape = [1 for _ in lr_ct.shape]

        epsilon = torch.rand(epsilon_shape, device=self._device).float()
        ct_hat = epsilon * lr_ct + (1 - epsilon) * hr_ct
        ct_hat = ct_hat.float()
        ct_hat = ct_hat.to(self._device)

        return ct_hat

    def compute_gradient_penalty(self, lr_ct, hr_ct):
        sr_ct = self.generator(lr_ct)
        ct_hat = self.generate_mixed_ct(sr_ct, hr_ct)
        ct_hat_critic = self.discriminator(ct_hat)

        grad_outputs = torch.ones_like(ct_hat_critic).to(self._device)
        gradients = autograd.grad(outputs=ct_hat_critic, inputs=ct_hat,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = self._lambda_coeff * torch.mean(
            (gradients.norm(2, dim=1) - 1) ** 2)

        return gradient_penalty

    def w_loss(self, lr_ct, hr_ct):
        sr_ct = self.generator(lr_ct)
        sr_critic = self.discriminator(sr_ct)
        hr_critic = self.discriminator(hr_ct)
        gp = self.compute_gradient_penalty(lr_ct, hr_ct)

        w_loss = -torch.mean(hr_critic) + torch.mean(sr_critic) + gp
        w_loss = (1 - self._alpha) * w_loss

        return w_loss

    def psnr(self, lr_ct, hr_ct):
        sr_ct = self.generator(lr_ct)
        criterion = nn.MSELoss()

        loss = criterion(sr_ct, hr_ct)
        loss = loss.data.cpu().numpy()
        psnr = 10 * np.log10(1 / loss)
        sr_range = [torch.min(sr_ct).detach().cpu().numpy().item(),
                    torch.max(sr_ct).detach().cpu().numpy().item()]

        return psnr, sr_range
