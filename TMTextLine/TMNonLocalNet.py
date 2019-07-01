import torch.nn as nn
from typing import Union,Optional
import torch
import torch.nn.functional as F
class NonLocalNet(nn.Module):
    def __init__(self,in_channels:int,inner_channels:Optional[int]=None,category:str='gaussion',bn:bool=False,subsample:bool=False):
        super(NonLocalNet,self).__init__()
        assert category in ['gaussion','dot_product'],'unexpected got:{}'.format(category)
        self.category=category
        self.in_channels=in_channels
        if inner_channels is not None:
            self.inner_channels=inner_channels
        else:
            self.inner_channels=in_channels
        self.theta=nn.Conv2d(self.in_channels,self.inner_channels,1,1)
        self.phi=nn.Conv2d(self.in_channels,self.inner_channels,1,1)
        self.g=nn.Conv2d(self.in_channels,self.in_channels,1,1)
        self.W=nn.Conv2d(self.inner_channels,self.in_channels,1,1)
        if bn:
            self.W=nn.Sequential(self.W,
                                 nn.BatchNorm2d(self.in_channels))
        if subsample:
            self.phi=nn.Sequential(self.phi,
                                   nn.MaxPool2d(2,2))
            self.g=nn.Sequential(self.g,
                                 nn.MaxPool2d(2,2))
    def forward(self, x:torch.Tensor):
        batch_size=x.size(0)
        theta=self.theta(x).reshape(batch_size,self.inner_channels,-1).permute(0,2,1)
        phi=self.phi(x).reshape(batch_size,self.inner_channels,-1)
        g=self.g(x).reshape(batch_size,self.inner_channels,-1).permute(0,2,1)
        theta_phi=torch.bmm(theta,phi)
        if self.category=='gaussion':
            theta_phi=F.softmax(theta_phi,dim=-1)
        else:
            theta_phi/=phi.size(-1)
        g=torch.bmm(theta_phi,g)
        g=g.permute((0,2,1)).reshape((batch_size,self.inner_channels,*x.size()[2:]))
        w=self.W(g)
        return w+x
if __name__=='__main__':
    a=NonLocalNet(3,3)
    b=torch.rand(2,3,100,100)
    c=a(b)
    print(a)

