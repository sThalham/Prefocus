import torch

from backbone import backbone_model
from model import CAOS_Model


def create_model():
    bb_model = backbone_model('resnext50')
    model = CAOS_Model(bb_model)
    return model