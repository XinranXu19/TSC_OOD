import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    def __init__(self,In_channel,Out_channel,downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = nn.Sequential(
            #in_channels=input_shape[0], out_channels=n_feature_maps, kernel_size=8, padding=4
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=In_channel, out_channels=Out_channel, kernel_size=8),
            nn.BatchNorm1d(Out_channel),
            nn.ReLU(),
            nn.Conv1d(Out_channel, Out_channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(Out_channel),
            nn.ReLU(),
            nn.Conv1d(Out_channel, Out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(Out_channel),
            #nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = nn.Sequential(
                nn.Conv1d(In_channel, Out_channel, kernel_size=1),
                nn.BatchNorm1d(Out_channel),
                #nn.ReLU()
            )
        else:
            self.res_layer = None

        self.relu = nn.ReLU()

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        out = self.layer(x)
        out += residual
        return self.relu(out)
        #return self.relu(self.layer(x)+residual)

class ResNet(torch.nn.Module):
    def __init__(self,in_channels=1,classes=5):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            Bottleneck(in_channels,64),
            Bottleneck(64,128),
            Bottleneck(128,128),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(128,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1, 128)
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    x = torch.randn(size=(3,1,10))
    model = ResNet(in_channels=1)

    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    #print(f'输入为:{x}')
    print(f'输出尺寸为:{output.shape}')
    #print(model)
    summary(model,(1,10),device='cpu')

