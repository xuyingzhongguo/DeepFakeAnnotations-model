from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from network.efficientb0 import EfficientNetB0, EfficientNetB0_features
# from network.convnext_tiny import Convnext_tiny
from dataset.transform import xception_default_data_transforms, xception_aug_transforms, mesonet_transforms, \
    effb0_transforms
from torchvision import models


# class EfficientNetB0_config_2(object):
#     model_name = 'effb0'
#     pretrained = True
#     model = EfficientNet(model_name=model_name, num_classes=2, pretrained=pretrained)
#
#     # transform
#     transform = eff_b0_transform
#
#     # SGD
#     weight_decay = 5e-4
#     momentum = 0.9
#
#     # learning rate
#     lr = 0.01
#
#     # others
#     print_freq = 100


class Xception_config(object):
    model_name = 'xception'
    # pretrained = False
    # model = xception(num_classes=2, pretrained=pretrained)
    # model = Xception_reconstructed(num_classes=2, pretrained=pretrained)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    lr = 0.01

    # transform
    transform = xception_aug_transforms


class Meso_Incp4():
    model_name = 'meso-in4'
    model = MesoInception4()
    # model = Meso4()
    # model = model_selection(modelname='mesonet', num_out_classes=2, dropout=0.5)
    lr = 0.001

    # transform
    transform = mesonet_transforms


class EffB0():
    model_name = 'effb0'
    model = EfficientNetB0(num_classes=2, pretrained=True)
    lr = 0.01

    # transform
    transform = effb0_transforms

class EffB0_features():
    model_name = 'effb0_features'
    model = EfficientNetB0_features(num_classes=2, pretrained=True)
    # transform
    transform = effb0_transforms

# class ConvNext():
#     model_name = 'convnext-tiny'
#     model = Convnext_tiny(num_classes=2, pretrained=True)
#     # model = models.convnext_tiny(num_classes=2, pretrained=False)
#     lr = 0.01
#
#     # transform
#     transform = convnext_tiny_transforms


# class Vit16():
#     from transformers import ViTForImageClassification
#
#     model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
#     lr = 0.01
#
#     # transform
#     transform = vit16_transforms


configs = {
    # 'effb0': EfficientNetB0_config_2(),
    'xception': Xception_config(),
    # 'vit16': Vit16_config_2(),
    'meso-in4': Meso_Incp4(),
    'effb0': EffB0(),
    'effb0_features': EffB0_features(),
    # "convnext-tiny": ConvNext(),
    # "vit-16": Vit16()
}


