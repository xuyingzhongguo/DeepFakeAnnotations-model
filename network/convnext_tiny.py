# import torchvision.models as models
# import torch.nn as nn
#
#
# class Convnext_tiny(nn.Module):
#     def __init__(self, num_classes=2, pretrained=True):
#         super().__init__()
#         model = models.convnext_tiny(pretrained=pretrained)
#
#         layers = list(model.children())[:-1]
#         fc_size = model.classifier[2].in_features
#
#         self.parent = nn.Sequential(*layers)
#         self.dropout = nn.Dropout(p=0.2, inplace=True)
#         self.fc = nn.Linear(in_features=fc_size, out_features=num_classes)
#
#     def forward(self, image):
#         x = self.parent(image)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x