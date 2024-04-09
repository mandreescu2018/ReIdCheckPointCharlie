import torch
import torchvision
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F


def weights_init_classifier(m):
    """
    This weight initialization strategy is commonly used to prevent the model from 
    getting stuck during training due to very small or zero weights. 
    Initializing weights with small random values helps to break the symmetry and allows the network 
    to learn more effectively.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def make_model(cfg, num_class, camera_num, view_num):
    model = BuildModel(num_class, camera_num, view_num, cfg)
    return model

class LinearRegressionModel(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights.to(self.device) * x + self.bias.to(self.device)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) 
        self.num_heads = num_heads

    def forward(self, x): 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) 

        return x 

class Bottleneck(nn.Module):
    # The expansion factor is set to 4, which means that the number of output channels in the final 
    # convolutional layer is 4 times the number of input channels. 
    # This expansion allows for richer representations in deeper layers.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        Parameters:
            inplanes (str): The number of input channels.
            planes: The number of output channels (before expansion).
            stride: The stride for the middle convolutional layer (default is 1).

        Returns:
            None. 
        """
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # conv1: A 1x1 convolution layer that reduces the number of channels (width) from inplanes to planes.
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        # Batch normalization layers that normalize the activations after each convolution to stabilize training.
        self.bn1 = nn.BatchNorm2d(planes)

        # conv2: A 3x3 convolution layer with optional striding (stride) to further process the features.
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        # Batch normalization layers that normalize the activations after each convolution to stabilize training.
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # conv3: A 1x1 convolution layer that expands the number of channels from planes to planes * expansion.
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        # Batch normalization layers that normalize the activations after each convolution to stabilize training.
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        # The input x is preserved as a residual connection, and it is passed through conv1 and bn1 
        # if either the stride is not 1 or the number of input channels does not match the number of output channels (due to expansion). 
        # This is done to ensure that the dimensions of the residual match the output dimensions.
        # So, identity <=> residual in fact
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        # The output of conv3 is added to the residual. 
        # This skip connection helps prevent vanishing gradients and allows for easier training of very deep networks.
        out += identity
        # Another ReLU activation function is applied to the combined output before returning it as the block's output.
        out = self.relu(out)
        return out

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x3 = self.layer3(x) 
        x4 = self.layer4(x3) 
        xproj = self.attnpool(x4) 

        return x3, x4, xproj 


class BuildModel(nn.Module):
    def __init__(self, camera_num, view_num, cfg) -> None:
        super(BuildModel, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        self.in_planes_proj = 1024
        self.num_classes = cfg.NUM_CLASSES
        self.camera_num = camera_num
        self.view_num = view_num

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        if cfg.MODEL.PRETRAIN_CHOICE != 'resume':
            self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        if cfg.MODEL.PRETRAIN_CHOICE != 'resume':
            self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        if cfg.MODEL.PRETRAIN_CHOICE != 'resume':
            self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        if cfg.MODEL.PRETRAIN_CHOICE != 'resume':
            self.bottleneck_proj.apply(weights_init_kaiming)

        h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)

        weights = torchvision.models.ResNet50_Weights.DEFAULT # DEFAULT mean the best available
        # resnet_model = torchvision.models.resnet50(weights=weights)
        resnet_model = ModifiedResNet(layers=(3, 4, 6, 3),
                output_dim=1024,
                heads=32,
                input_resolution=h_resolution*w_resolution,
                width=64)
        self.image_encoder = resnet_model

    
    def forward(self, x, label=None, cam_label= None, view_label=None):
        if self.model_name == 'RN50':
            output_size = (16, 8)

            image_features_last, image_features, image_features_proj = self.image_encoder(x) #B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            # img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, output_size).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) #B,512  B,128,512
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
            
    def load_param_resume(self, model_path, optimizer, scheduler):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        return self, optimizer, checkpoint['epoch'], scheduler, loss



