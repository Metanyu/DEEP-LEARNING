import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class ECCVGenerator(BaseColor):
    
    def __init__(self, num_classes=313, norm_layer=nn.BatchNorm2d, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [norm_layer(64)]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [norm_layer(128)]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [norm_layer(256)]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [norm_layer(512)]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [norm_layer(512)]

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [norm_layer(512)]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [norm_layer(512)]

        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(num_classes, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, input_l):
        input_l_paper = (input_l + 1.0) * 50.0
        
        conv1_2 = self.model1(self.normalize_l(input_l_paper))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        
        if self.dropout is not None:
            conv4_3 = self.dropout(conv4_3)
        
        conv5_3 = self.model5(conv4_3)
        if self.dropout is not None:
            conv5_3 = self.dropout(conv5_3)
            
        conv6_3 = self.model6(conv5_3)
        if self.dropout is not None:
            conv6_3 = self.dropout(conv6_3)
            
        conv7_3 = self.model7(conv6_3)
        if self.dropout is not None:
            conv7_3 = self.dropout(conv7_3)
            
        conv8_3 = self.model8(conv7_3)
        return conv8_3
    
    def forward_to_ab(self, input_l):
        conv8_3 = self.forward(input_l)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))


class ColorizationModel(nn.Module):
    
    def __init__(self, output_type='classification', num_classes=313, dropout=0.0):
        super().__init__()
        self.output_type = output_type
        self.num_classes = num_classes
        
        if output_type == 'classification':
            self.model = ECCVGenerator(num_classes=num_classes, dropout=dropout)
        else:
            self.model = ECCVGenerator(num_classes=num_classes, dropout=dropout)
            self.regression_head = nn.Sequential(
                nn.Softmax(dim=1),
                nn.Conv2d(num_classes, 2, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Tanh()
            )
    
    def forward(self, x):
        logits = self.model(x)
        
        if self.output_type == 'classification':
            return logits
        else:
            return self.regression_head(logits)


class ClassificationToAB(nn.Module):
    
    def __init__(self, ab_gamut, temperature=0.38):
        super().__init__()
        self.temperature = temperature
        self.register_buffer('ab_centers',
                            torch.from_numpy(ab_gamut.ab_gamut).float())
    
    def forward(self, logits):
        B, Q, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        annealed = torch.exp(log_probs / self.temperature)
        annealed = annealed / (annealed.sum(dim=1, keepdim=True) + 1e-8)  # (B, Q, H, W)
        
        # Weighted sum of ab centers
        # annealed: (B, Q, H, W), ab_centers: (Q, 2)
        annealed = annealed.permute(0, 2, 3, 1)  # (B, H, W, Q)
        ab = torch.matmul(annealed, self.ab_centers)  # (B, H, W, 2)
        
        return ab.permute(0, 3, 1, 2)  # (B, 2, H, W)
