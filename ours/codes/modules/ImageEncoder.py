
import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
	def __init__(self, non_trainable=True):
		super(ResNet, self).__init__()
		self.resnet152 = models.resnet152(pretrained=True)
		if non_trainable:
			for param in self.resnet152.parameters():
				param.requires_grad = False
		self.select = 'layer4' #conv5

	def forward(self, x):
		for name, layer in self.resnet.named_children():
			x = layer(x)
			if name in self.select:
				break
		return x

class SPool(nn.Module):
	def __init__(self):
		super(SPool, self).__init__()

	def forward(self, x):
		max_x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
		min_x = torch.min(torch.min(x, dim=2)[0], dim=2)[0]
		x = max_x + min_x
		return x

class ImageEncoder(nn.Module):
  def __init__(self, input_size, D=2048, D_prime=2400):
    super(ImageEncoder, self).__init__()
    self.resnet = ResNet(non_trainable=True)
    self.conv1x1 = nn.Conv2d(D, D_prime, 1)
    self.spool = SPool()
    self.proj = nn.Linear(D_prime, D_prime)
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
  	x = self.resnet(x)
  	x = self.conv1x1(x)
  	x = self.spool(x)
  	x = self.dropout(x)
  	x = self.proj(x)
  	x = x / torch.norm(x, 2, dim=2)

    return x


