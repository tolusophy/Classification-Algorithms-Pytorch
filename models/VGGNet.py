import torch
import torch.nn as nn

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG_net(nn.Module):
    '''
    arch        : VGG11, VGG13, VGG16 or VGG19 from VGG_types
    in_channels : number of channels in the input image
    num_classes : number of outputs in the final layer
    '''
    def __init__(self, arch, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[arch])
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        '''
        :param architecture: The list of conv operations and MP in order of their operation
        :return:  returns nn.Sequential(all conv operations)
        '''
        layers = []
        in_channels = self.in_channels   # number of channels in the input image

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=3, stride=1, padding=1),                 # Same Conv
                           nn.BatchNorm2d(out_channels),  ## not in original paper
                           nn.ReLU()]

                in_channels = out_channels
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    vgg = VGG_net('VGG11', in_channels=3, num_classes=10)
    print(vgg(x))