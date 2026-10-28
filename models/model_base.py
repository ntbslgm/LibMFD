import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(ClassifierMLP, self).__init__()
        
        self.last = last
        self.net = nn.Sequential(
                   nn.Dropout(p=dropout),
                   
                   nn.Linear(input_size, int(input_size/4)),
                   nn.ReLU(),
                   
                   nn.Linear(int(input_size/4), int(input_size/16)),
                   nn.ReLU(),
                   
                   nn.Linear(int(input_size/16), output_size))
        
        if last == 'logsm':
            self.last_layer = nn.LogSoftmax(dim=-1)
        elif last == 'sm':
            self.last_layer = nn.Softmax(dim=-1)
        elif last == 'tanh':
            self.last_layer = nn.Tanh()
        elif last == 'sigmoid':
            self.last_layer = nn.Sigmoid()
        elif last == 'relu':
            self.last_layer = nn.ReLU()

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)
        
        return y


class CNNlayer(nn.Module):

    def __init__(self,
                 in_channel=1, kernel_size=8, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(CNNlayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        dp = nn.Dropout(dropout)
        
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            dp,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        
        return h


class CNNFeatureExtractor(nn.Module):

    def __init__(self, in_channel, window_sizes=[4, 8, 16, 24, 32], block=CNNlayer, dropout=0.):
        super(CNNFeatureExtractor, self).__init__()
       
        self.convs = nn.ModuleList([
                       block(in_channel=in_channel, kernel_size=h, dropout=dropout)
                       for h in window_sizes])
                              
        self.fl = nn.Flatten()

    def forward(self, input):
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = self.fl(out)
        
        return out


class BasicBlock1D(nn.Module):
    """
    Basic residual block for the 1D ResNet backbone.
    Two 1D convolutions with a residual (skip) connection.
    """

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet feature extractor for vibration signals.

    Output feature dimension is projected to 2560 to match the default
    ``FeatureExtractor`` (multi-scale 1D-CNN), so all downstream classifiers
    and domain discriminators remain unchanged regardless of the backbone.
    """

    def __init__(self, in_channel=1, dropout=0.):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 2560)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        layers = [BasicBlock1D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


_BACKBONE = 'CNN'


def set_backbone(name):
    """Set the global backbone used by the ``FeatureExtractor`` factory."""
    global _BACKBONE
    if name not in ('CNN', 'ResNet'):
        raise ValueError(f"Unknown backbone: {name} (expected 'CNN' or 'ResNet')")
    _BACKBONE = name


def FeatureExtractor(in_channel, window_sizes=[4, 8, 16, 24, 32], block=CNNlayer, dropout=0.):
    """
    Factory returning a feature extractor of the globally-selected backbone.

    Both backbones output a 2560-dim feature vector so all downstream
    classifiers and domain discriminators remain unchanged. The backbone is
    chosen via :func:`set_backbone` (driven by the ``model_name`` suffix in
    train.py), so model files do not need to be modified.
    """
    if _BACKBONE == 'ResNet':
        return ResNet1D(in_channel=in_channel, dropout=dropout)
    return CNNFeatureExtractor(in_channel=in_channel, window_sizes=window_sizes,
                               block=block, dropout=dropout)


class BaseModel(nn.Module):

    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()

        self.G = FeatureExtractor(in_channel=input_size, dropout=dropout)

        self.C = ClassifierMLP(2560, num_classes, dropout, last=None)

    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        if self.training:
            return predictions, f
        else:
            return predictions
