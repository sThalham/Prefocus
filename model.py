import torch.nn as nn
import torch.nn.functional as F


class CAOS_Model(nn.Module):
    def __init__(self, backbone_model, num_classes, box_points):
        super(CAOS_Model, self).__init__()
        self.backbone = backbone_model()
        self.FPN = FeaturePyramid()
        self.dis = FBDiscriminator()
        self.cls = ClassificationHead(num_classes)
        self.box = BoxHead(box_points)
        self.cub = CuboidHead(8)

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        P3, P4, P5, P6, P7 = self.FPN(C3, C4, C5)

        P3_size = P3.size()
        P4_size = P4.size()
        P5_size = P5.size()
        P6_size = P6.size()
        P7_size = P7.size()

        # discriminator
        dis_P3 = self.dis(P3)
        dis_P3 = dis_P3.view(P3_size[0], P3_size[1], P3_size[2] * P3_size[3])


class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FBDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, 3)
        self.conv2 = nn.Conv2d(512, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.conv5 = nn.Conv2d(64, 1, 3)

    def forward(self, P3, P4, P5):
        #x = x.view(-1, 16 * 4 * 4)
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        x = F.mish(self.fc4(x))
        x = F.sigmoid(self.fc5(x))

        return x


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