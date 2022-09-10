import torch

class backbone_model():

    def __init__(self, backbone_model):

        if backbone_model=='resnext50':
            self.model = self.loadrn50()
        elif backbone_model=='resnext101':
            self.model = self.loadrn101()
        else:
            print('Unrecognized backbone specified')

    def loadrn50(self):
        weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNeXt50_32X4D_Weights.IMAGENET1K_V2")
        model = torch.hub.load("pytorch/vision", "resnext50", weights=weights)

        return model

    def loadrn101(self):
        weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNeXt101_64X4D_Weights.IMAGENET1K_V1")
        model = torch.hub.load("pytorch/vision", "resnext101", weights=weights)

        return model

