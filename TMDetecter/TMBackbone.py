import torch.nn as nn
import torch.nn.functional as F
class Bottleneck(nn.Module):#[3, 4, 6, 3]:50;[3, 4, 23, 3]:101;[3, 8, 36, 3]:152
    expansion=2#4
    def __init__(self,inplanes,planes,stride=1,downsample=None,se=None):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,1,bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(planes,planes,3,stride=stride,padding=1,bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))
        self.conv3=nn.Sequential(nn.Conv2d(planes,planes*self.expansion,1,bias=False),
                                 nn.BatchNorm2d(planes*self.expansion),
                                 )
        self.relu=nn.ReLU(inplace=True)
        #self.se=se(planes*self.expansion,16)
        self.downsample=downsample
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        #if self.se is not None:
            #output=self.se(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
            #print('downsample shortcut size:{}'.format(shortcut.size()))
        output+=shortcut
        return self.relu(output)
class TMBackbone(nn.Module):
    def __init__(self,block=Bottleneck,nums_block_list:list=[3, 4, 6, 3],fpn=True):
        super(TMBackbone,self).__init__()
        self.inplanes = 64
        self.fpn=fpn
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64),#nn.GroupNorm(64 // 32, 64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64),#nn.GroupNorm(64 // 32, 64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, 2, 1, bias=False))  # nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, nums_block_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, nums_block_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nums_block_list[2], stride=2)
        if self.fpn==True:
            self.layer4 = self._make_layer(block, 512, nums_block_list[3], stride=2)
            # lateral layers
            self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)  # 2048
            self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)  # 1024
            self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)  # 512
            #top-down layers
            self.toplayer1 = nn.Conv2d(256, 256, 3, 1, 1)
            self.toplayer2 = nn.Conv2d(256, 256, 3, 1, 1)
            self.toplayer3 = nn.Conv2d(256, 256, 3, 1, 1)
        #self.latlayer=nn.Sequential(nn.Conv2d(self.inplanes,256,3,1,1))
    def forward(self, input):
        output = self.conv1(input)  # /2
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)  # /4
        output = self.layer1(output)#c2
        c3 = self.layer2(output)  # /8
        c4 = self.layer3(c3)  # /16
        if self.fpn==True:
            c5 = self.layer4(c4)#32
            p5 = self.latlayer1(c5)
            p5 = self.toplayer1(p5)
            p4 = self._upsample_add(p5,self.latlayer2(c4))
            p4 = self.toplayer2(p4)
            p3 = self._upsample_add(p4,self.latlayer3(c3))
            p3 = self.toplayer3(p3)
            return p3,p4,p5
        else:
            return [c4]
    def _make_layer(self,block,inplanes,nums_block,kernel_size=1,stride=1,padding=1):
        downsample=None
        print('inplanes:{}'.format(self.inplanes))
        if stride!=1 or self.inplanes!=inplanes*block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,inplanes*block.expansion,1,stride,bias=False),
                                     nn.BatchNorm2d(inplanes*block.expansion))#nn.GroupNorm(inplanes*block.expansion//32,inplanes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,inplanes,stride,downsample))
        self.inplanes=inplanes*block.expansion
        for _ in (1,nums_block):
            layers.append(block(self.inplanes,inplanes))
        return nn.Sequential(*layers)
    def _upsample_add(self,x,y):
        _,_,h,w=y.size()
        return F.interpolate(x,size=(h,w),mode='bilinear')+y