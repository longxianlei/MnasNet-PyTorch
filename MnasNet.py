from torch.autograd import Variable
import torch.nn as nn
import torch


def Conv_3x3(input, output, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(output, momentum=0.01),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(input, output):
    return nn.Sequential(
        nn.Conv2d(input, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output, momentum=0.01),
        nn.ReLU6(inplace=True)
    )

def SepConv_3x3(input, output): # input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(input, input, 3, 1, 1, groups=input, bias=False),
        nn.BatchNorm2d(input, momentum=0.01),
        nn.ReLU6(inplace=True),
        # pw + linear
        nn.Conv2d(input, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output, momentum=0.01),
    )

class InvertedResidual(nn.Module):
    def __init__(self, input, output, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and input == output

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(input, input*expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input*expand_ratio, momentum=0.01),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(input*expand_ratio, input*expand_ratio, kernel, stride, kernel//2,
                      groups=input*expand_ratio, bias=False),
            nn.BatchNorm2d(input*expand_ratio, momentum=0.01),
            nn.ReLU6(inplace=True),
            # pw + linear
            nn.Conv2d(input*expand_ratio, output, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output, momentum=0.01)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            return out + x
        else:
            return out


class MnasNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MnasNet, self).__init__()

        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # exp_r, out_c, repeat_n, stride, kernel
            [3, 24, 3, 2, 3],  # -->56x56x24
            [3, 40, 3, 2, 5],  # -->28x28x40
            [6, 80, 3, 2, 5],  # -->14x14x80
            [6, 96, 2, 1, 3],  # -->14x14x96
            [6, 192, 4, 2, 5],  # -->7x7x192
            [6, 320, 1, 1, 3],  # -->7x7x320
        ]

        assert input_size % 32 == 0
        input_channel = int(32*width_mult)
        self.last_channel = int(1280*width_mult) if width_mult > 1.0 else 1280

        # building the first 2 layers.
        self.features = [Conv_3x3(3, input_channel, 2), SepConv_3x3(input_channel, 16)]
        input_channel = int(16*width_mult)

        # building inverted residual blocks.
        for t, c, n, s, k in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building the last serval features.
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # build the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = MnasNet()
    x_image = Variable(torch.randn(1, 3, 224, 224))

    x = torch.randn(10, 3, 224, 224)
    # y = net(x)
    # print(y.size())

    y = net(x)
    print(y.size())

