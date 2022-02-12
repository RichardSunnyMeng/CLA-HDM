import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

class sub_model(nn.Module):
    def __init__(self):
        super(sub_model, self).__init__()
        self.us = resnet50(pretrained=True)
        self.us.fc = nn.Identity()

        self.cdfi = resnet50(pretrained=True)
        self.cdfi.fc = nn.Identity()

        self.GB = nn.AdaptiveAvgPool2d((1, 1))
        self.cdfiAtt = nn.Sequential(
            nn.Linear(3, 10, bias=False),
            nn.ReLU(),
            nn.Linear(10, 3, bias=False),
            nn.Softmax(-1),
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(-1),
        )

        self.fc_cdfi = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(-1),
        )

        self.fc_us = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(-1),
        )

    def forward(self, x_cdfi, x_us):
        cdfi_weight = self.cdfiAtt(self.GB(x_cdfi).reshape(-1, 3)).reshape(-1, 3, 1, 1)
        x_cdfi_weighted = torch.mul(x_cdfi, cdfi_weight)

        y_cdfi = self.cdfi(x_cdfi_weighted) # B * 512
        y_us = self.us(x_us) # B * 512

        o_cdfi = self.fc_cdfi(y_cdfi)
        o_us = self.fc_us(y_us)

        y = torch.cat((y_us, y_cdfi), 1)

        o = self.fc_fusion(y)

        return o, o_cdfi, o_us


class CLA_HDM(nn.Module):
    """
            The users can train CLA-HDM end-to-end by "self.forward".
            However in our work, we directly output the final probabilities
        based on the sub-models trained separately instead of fine-tuning,
        as seen in "self.forward_inference".

        x_cdfi: tensor, (B * C * W * H) and B = 1 in forward_inference while B can be a Integer in forward.
        x_us: tensor, (B * C * W * H) and B = 1 in forward_inference while B can be a Integer in forward.
    """
    def __init__(self, thr=0.5):
        super(CLA_HDM, self).__init__()
        self.sub_model1 = sub_model()
        self.sub_model2 = sub_model()
        self.sub_model3 = sub_model()
        self.load_submodel_dict()

        self.thr = thr

    def load_submodel_dict(self):
        pass

    def forward_inference(self, x_cdfi, x_us):
        with torch.no_grad():
            y1 = self.sub_model1(x_cdfi, x_us)
            if y1[1] > self.thr:
                y = self.sub_model3(x_cdfi, x_us)
                y = torch.cat([torch.zeros_like(y), y], dim=-1)
            else:
                y = self.sub_model2(x_cdfi, x_us)
                y = torch.cat([y, torch.zeros_like(y)], dim=-1)
        return y

    def forward(self, x_cdfi, x_us):
        y1 = self.sub_model1(x_cdfi, x_us)

        y2 = self.sub_model2(x_cdfi, x_us)
        y3 = self.sub_model3(x_cdfi, x_us)

        y2 = y2 * y1[0:1]
        y3 = y3 * y1[1:]
        return torch.cat([y2, y3], dim=-1)






