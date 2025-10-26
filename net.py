import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numpy import prod
import config
from transformers import ViTForImageClassification

class SiameseViTNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained vit-base-patch16-224.
    '''
    def __init__(self):
        super(SiameseViTNet, self).__init__()
        pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(768, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),

                                nn.Linear(512, 64),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(64, 8))
        
    def forward_once(self, x, return_patches=False):
        output = self.model_conv(x)
        last_hidden = output.last_hidden_state  # shape: (B, 1+N, 768)
        if return_patches:
            patch_embeddings = last_hidden[:, 1:, :]  # shape: (B, N, 768)
            return patch_embeddings  # Not sent in fc
        else:
            cls_token = last_hidden[:, 0, :]  # shape: (B, 768)
            output = self.fc(cls_token)       # shape: (B, 8)
            return output

    def forward(self, input1, input2, return_patches=False):
        output1 = self.forward_once(input1, return_patches=return_patches)
        output2 = self.forward_once(input2, return_patches=return_patches)
        return output1, output2

class SiameseEfficientNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained EfficientNet-B0.
    '''
    def __init__(self):
        super(SiameseEfficientNet, self).__init__()
        pretrained_model = torchvision.models.efficientnet_b0(pretrained=True)
        self.model_conv = pretrained_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.2),
            nn.PReLU(1),
            nn.Linear(64, 8)
        )

    def forward_once(self, x):
        output = self.model_conv(x)               
        output = self.pool(output)                   
        output = output.view(output.size(0), -1)          
        output = self.fc(output)
        return output

    def forward(self, input1, input2, return_patches=False):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class SiameseMobileNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained MobileNet.
    '''
    def __init__(self):
        super(SiameseMobileNet, self).__init__()
        pretrained_model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model_conv = pretrained_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.2),
            nn.PReLU(1),
            nn.Linear(64, 8)
        )

    def forward_once(self, x):
        output = self.model_conv(x)               
        output = self.pool(output)                   
        output = output.view(output.size(0), -1)          
        output = self.fc(output)
        return output

    def forward(self, input1, input2, return_patches=False):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2    

class SiameseResNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained ResNet.
    '''

    def __init__(self):
        super(SiameseResNet, self).__init__()
        pretrained_model = torchvision.models.resnet34(pretrained=True)
        if config.RESNET_POOLING == 'fixed' and str(pretrained_model.avgpool)[:8] == 'Adaptive':
            pretrained_model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(prod(config.RES34_960x720_SHAPE), 2048),
                                nn.BatchNorm1d(2048),
                                nn.ReLU(inplace=True),

                                nn.Linear(2048, 512),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(512, 8))

    def forward_once(self, x):
        output = self.model_conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2, return_patches=False):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseSqueezeNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained SqueezeNet.
    '''

    def __init__(self):
        super(SiameseSqueezeNet, self).__init__()
        pretrained_model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.reduce_dim = nn.Conv2d(512, 16, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Sequential(nn.Linear(16 * prod(config.SQUEEZE_960x720_SHAPE), 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),

                                nn.Linear(512, 64),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(64, 8))

    def forward_once(self, x):
        output = self.model_conv(x)
        output = self.reduce_dim(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2, return_patches=False):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Copied from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
        output1, output2 = outputs[0], outputs[1]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive