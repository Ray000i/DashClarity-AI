"""
Zero-DCE and Zero-DCE++ Model Architectures
Lightweight, zero-reference deep curve estimation for low-light enhancement.
Reference:
- Zero-DCE: Li et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement", TPAMI 2021.
- Zero-DCE++: Li et al., "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation", BMVC 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """Depthwise Separable Convolution block used in Zero-DCE++."""
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ZeroDCEPlus(nn.Module):
    """
    Zero-DCE++ architecture (~10k parameters).
    Outputs a 3-channel illumination curve parameter map applied iteratively.
    """
    def __init__(self, scale_factor=1.0):
        super(ZeroDCEPlus, self).__init__()
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, A):
        """
        Iterative quadratic curve estimation for 8 iterations:
        LE_n(x) = LE_{n-1}(x) + A * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        """
        enh = x
        for _ in range(8):
            enh = enh + A * (torch.pow(enh, 2) - enh)
        return enh

    def forward(self, x):
        if self.scale_factor == 1.0:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        A = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor != 1.0:
            A = F.interpolate(A, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        enhanced = self.enhance(x, A)
        return enhanced, A


class ZeroDCE(nn.Module):
    """
    Standard Zero-DCE architecture (~79k parameters).
    """
    def __init__(self, num_filters=32):
        super(ZeroDCE, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, num_filters, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(num_filters * 2, 24, 3, 1, 1, bias=True)

    def enhance(self, x, A):
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(A, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        x = x + r8 * (torch.pow(x, 2) - x)
        return x

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        A = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        enhanced = self.enhance(x, A)
        return enhanced, A
