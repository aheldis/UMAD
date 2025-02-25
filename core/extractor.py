import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x



class SRSmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, dx=0, dy=0):
        super(SRSmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # SR parameters
        self.dx = dx
        self.dy = dy
        self.patch_size = 8
        self.patch_pixels = self.patch_size ** 2

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # SR starts here
        b, _, h, w = x.shape # b (batch_size) * c (3) * h (32) * w (32)
        
        # create list of augmented images
        aug_list = []
        ind_i_j = {}
        ind = 0
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                aug_list.append(transforms.functional.affine(x, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1
    
        # concatenate all augmented images into a single batch
        x = torch.cat(aug_list, dim=0)

        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_field = self.conv2(x)

        # init with first feature
        feature_field_ensembled = feature_field[0:b, :, :, :].clone()

        # for each shifted feature
        for k in range(1, ind):
            i, j = ind_i_j[k]
            h_mag = abs(i)
            w_mag = abs(j)

            feature_field_temp = feature_field[k*b:(k+1)*b, :, :, :]
            feature_field_ensembled += feature_field_temp * ((self.patch_size - h_mag) * (self.patch_size - w_mag) / self.patch_pixels)
         
            if i > 0:
                feature_field_ensembled[:,:,:-1,:] += feature_field_temp[:,:,1:,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)
            elif i < 0:
                feature_field_ensembled[:,:,1:,:] += feature_field_temp[:,:,:-1,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)

            if j > 0:
                feature_field_ensembled[:,:,:,:-1] += feature_field_temp[:,:,:,1:]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)
            elif j < 0:
                feature_field_ensembled[:,:,:,1:] += feature_field_temp[:,:,:,:-1]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)

            if i > 0 and j > 0:
                feature_field_ensembled[:,:,:-1,:-1] += feature_field_temp[:,:,1:,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i > 0 and j < 0:
                feature_field_ensembled[:,:,:-1,1:] += feature_field_temp[:,:,1:,:-1]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j > 0:
                feature_field_ensembled[:,:,1:,:-1] += feature_field_temp[:,:,:-1,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j < 0:
                feature_field_ensembled[:,:,1:,1:] += feature_field_temp[:,:,:-1,:-1]*(h_mag*w_mag / self.patch_pixels)  

        x = feature_field_ensembled

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SRBasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, dx=1, dy=1):
        super(SRBasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        # SR parameters
        self.dx = dx
        self.dy = dy
        self.patch_size = 8
        self.patch_pixels = self.patch_size ** 2
        self.avgpool_srt = nn.AvgPool2d(8)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # SR starts here
        aug_list = []
        b, _, h, w = x.shape # b (batch_size) * c (3) * h  * w

        ind_i_j = {}
        ind = 0

        # shifting the image by i, j, and append
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                aug_list.append(transforms.functional.affine(x, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1
        image_batch = torch.cat(aug_list, dim=0)   #stacking them into one batch: (n*b) * 3 * 32 * 32 

        w = w * 8
        h = h * 8

        x = image_batch

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feature_field = self.conv2(x)


        # Encoded (features) (n*b) * c (512) * 4 * 4
        feature_field_full = torch.nn.functional.interpolate(feature_field[0:b, :, :, :], scale_factor=8, mode='nearest')
        # feature_field_full: (n*b) * c (512) * 32 * 32

        # inverse shifts
        for k in range(ind):
            if k == 0:
                continue
            i, j = ind_i_j[k]
            new_feature_field = torch.nn.functional.interpolate(feature_field[k*b:k*b+b, :, :, :], scale_factor=8, mode='nearest')
            # remark: Rolling average
            feature_field_full += transforms.functional.affine(new_feature_field, translate=[-i,-j], angle=0, scale=1, shear=0)

        feature_fine_mean = feature_field_full / ind
        x = self.avgpool_srt(feature_fine_mean)


        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
'''        
        # create list of augmented images
        aug_list = []
        ind_i_j = {}
        ind = 0
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                aug_list.append(transforms.functional.affine(x, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1
    
        # concatenate all augmented images into a single batch
        x = torch.cat(aug_list, dim=0)


        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feature_field = self.conv2(x)

        # init with first feature
        feature_field_ensembled = feature_field[0:b, :, :, :].clone()

        # for each shifted feature
        for k in range(1, ind):
            i, j = ind_i_j[k]
            h_mag = abs(i)
            w_mag = abs(j)

            feature_field_temp = feature_field[k*b:(k+1)*b, :, :, :]
            feature_field_ensembled += feature_field_temp * ((self.patch_size - h_mag) * (self.patch_size - w_mag) / self.patch_pixels)
         
            if i > 0:
                feature_field_ensembled[:,:,:-1,:] += feature_field_temp[:,:,1:,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)
            elif i < 0:
                feature_field_ensembled[:,:,1:,:] += feature_field_temp[:,:,:-1,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)

            if j > 0:
                feature_field_ensembled[:,:,:,:-1] += feature_field_temp[:,:,:,1:]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)
            elif j < 0:
                feature_field_ensembled[:,:,:,1:] += feature_field_temp[:,:,:,:-1]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)

            if i > 0 and j > 0:
                feature_field_ensembled[:,:,:-1,:-1] += feature_field_temp[:,:,1:,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i > 0 and j < 0:
                feature_field_ensembled[:,:,:-1,1:] += feature_field_temp[:,:,1:,:-1]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j > 0:
                feature_field_ensembled[:,:,1:,:-1] += feature_field_temp[:,:,:-1,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j < 0:
                feature_field_ensembled[:,:,1:,1:] += feature_field_temp[:,:,:-1,:-1]*(h_mag*w_mag / self.patch_pixels)  

        x = feature_field_ensembled
'''