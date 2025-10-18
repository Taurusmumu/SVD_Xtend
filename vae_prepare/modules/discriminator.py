import functools
import torch.nn as nn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class PatchGAN3D(nn.Module):

    """
    PatchGAN discriminator as defined in Image to Image Translation w/ Conditional Adversarial Networks
    https://arxiv.org/pdf/1611.07004 
    """

    def __init__(self, 
                 input_channels=3, 
                 start_dim=64, 
                 depth=3,
                 kernel_size=(3, 3, 3),
                 padding=(1, 1, 1),
                 stride=(1, 2, 2),
                 leaky_relu_slope=0.2):
        
        super(PatchGAN3D, self).__init__()

        current_filters = start_dim
        layers = nn.ModuleList([])

        ### Projection from input_channels to start_dim ###
        layers.append(nn.Conv3d(input_channels, current_filters, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.LeakyReLU(leaky_relu_slope))

        ### Loop For All the Next Layes ###
        for i in range(depth):

            ### Apply a stride of 2 on all convoutions except the last ###
            stride = (1, 2, 2) if i != depth-1 else (1, 1, 1)
            out_channels = current_filters * 2
            layers.append(nn.Conv3d(current_filters, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            ### Update Current Filters ###
            current_filters = out_channels

        # Output will have a single channel
        layers.append(nn.Conv3d(current_filters, 1, kernel_size=kernel_size, stride=1, padding=padding))
        
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

# def init_weights(module):
#     if isinstance(module, nn.Conv2d):
#         nn.init.normal_(module.weight.data, 0.0, 0.2)
#     elif isinstance(module, nn.BatchNorm2d):
#         nn.init.normal_(module.weight.data, 1.0, 0.02)
#         nn.init.constant_(module.bias.data, 0.0)