import torch
import torch.nn as nn
import torch.nn.functional as F


from BiFPN import BiFPN

class Prefocus_Model(nn.Module):
    def __init__(self, backbone_model, num_classes):
        super(Prefocus_Model, self).__init__()
        self.backbone = backbone_model()
        self.FPN = BiFPN([sizes])
        self.dis = FBDiscriminator()
        self.cls = ClassificationHead(num_classes)
        self.box = BoxHead(4)
        self.cub = CuboidHead(8)

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        P3, P4, P5, P6, P7 = self.FPN(C3, C4, C5)

        maps = [P3, P4, P5, P6, P7]
        fm_sizes = [P3.size(), P4.size(), P5.size(), P6.size(), P7.size()]

        locs = []
        clss = []
        boxs = []
        cubs = []
        for lv, map in enumerate(maps):
            #discriminator
            dis_level = self.dis(map)
            dis_level = dis_level.view(fm_sizes[lv][0], fm_sizes[lv][1], fm_sizes[lv][2] * fm_sizes[lv][3])
            locs.append(dis_level)
            # classes
            cls_level = self.cls(map)
            cls_level = cls_level.view(fm_sizes[lv][0], fm_sizes[lv][1], fm_sizes[lv][2] * fm_sizes[lv][3])
            clss.append(cls_level)
            # boxes
            box_level = self.box(map)
            box_level = box_level.view(fm_sizes[lv][0], fm_sizes[lv][1], fm_sizes[lv][2] * fm_sizes[lv][3])
            boxs.append(box_level)
            # cuboids
            cub_level = self.cub(map)
            cub_level = cub_level.view(fm_sizes[lv][0], fm_sizes[lv][1], fm_sizes[lv][2] * fm_sizes[lv][3])
            cubs.append(cub_level)

        locations = torch.cat(locs, dim=2)
        classes = torch.cat(clss, dim=2)
        boxes = torch.cat(boxs, dim=2)
        cuboids = torch.cat(cubs, dim=2)

        return [locations, classes, boxes, cuboids]


class FBDiscriminator(nn.Module):
    def __init__(self):
        super(FBDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, 3)
        self.conv2 = nn.Conv2d(512, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.conv5 = nn.Conv2d(64, 1, 3)

    def forward(self, x):
        #x = x.view(-1, 16 * 4 * 4)
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        x = F.mish(self.fc4(x))
        x = F.sigmoid(self.fc5(x))

        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv5 = nn.Conv2d(256, num_classes, 3)

    def forward(self, x):
        # x = x.view(-1, 16 * 4 * 4)
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        x = F.mish(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x


class BoxHead(nn.Module):
    def __init__(self, reg_values):
        super(BoxHead, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3)
        self.conv2 = nn.Conv2d(512, 512, 3)
        self.conv3 = nn.Conv2d(512, 512, 3)
        self.conv4 = nn.Conv2d(512, 512, 3)
        self.conv5 = nn.Conv2d(512, reg_values, 3)

    def forward(self, x):
        # x = x.view(-1, 16 * 4 * 4)
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        x = F.mish(self.fc4(x))
        x = self.fc5(x)
        return x


class CuboidHead(nn.Module):
    def __init__(self, reg_values):
        super(CuboidHead, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3)
        self.conv2 = nn.Conv2d(512, 512, 3)
        self.conv3 = nn.Conv2d(512, 512, 3)
        self.conv4 = nn.Conv2d(512, 512, 3)
        self.conv5 = nn.Conv2d(512, reg_values, 3)

    def forward(self, x):
        # x = x.view(-1, 16 * 4 * 4)
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        x = F.mish(self.fc4(x))
        x = self.fc5(x)
        return x