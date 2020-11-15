import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetLinCLS(nn.Module):

    def __init__(self, base_model, num_classes):
        super(ResNetLinCLS, self).__init__()
        
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=num_classes),
                            "resnet34": models.resnet34(pretrained=False, num_classes=num_classes),
                            "resnet50": models.resnet50(pretrained=False, num_classes=num_classes)}

        resnet = self._get_basemodel(base_model)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(list(resnet.children())[-1].weight.shape[1], num_classes, bias=True)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
                                      
        return x
