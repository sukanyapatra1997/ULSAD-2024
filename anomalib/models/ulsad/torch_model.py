#----------------------------------------------------------------------------
# Created By  : Anomymous Author
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains Torch Model for the ULSAD algorithm.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from anomalib.models.components import FeatureExtractor
from anomalib.models.components import GaussianBlur2d

logger = logging.getLogger(__name__)

def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm

class PretrainExtractor(nn.Module):

    def __init__(self,
                 backbone: str,
                 layers: list[str],
                 pre_trained: bool = True,
                 target_dim: int = 384,
                 ):
        super().__init__()
        self.target_dim = target_dim
        self.encoder = FeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers)

    def forward(self, images: Tensor):

        images = imagenet_norm_batch(images)

        encoder_features = self.encoder(images)
        x1, x2 = encoder_features["layer2"], encoder_features["layer3"]

        b,c,h,w = x1.shape
        x2 = F.interpolate(x2, size=(h,w), mode="bilinear", align_corners=False)

        features = torch.cat([x1,x2],dim=1)
        b,c,h,w = features.shape
        features = features.reshape(b,c,h*w)
        features = features.transpose(1,2)

        target_features = F.adaptive_avg_pool1d(features, self.target_dim)
        target_features = target_features.transpose(1,2)
        target_features = target_features.reshape(b,self.target_dim,h,w)

        return target_features

class FeatAE(nn.Module):

    def __init__(self, in_dim=384, out_dim=1536, double_channel=False):
        super().__init__()

        #encoder
        self.conv1 = nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1 = nn.BatchNorm2d(out_dim//2)

        self.conv2 = nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn3 = nn.BatchNorm2d(out_dim)

        #decoder
        self.deconv1 = nn.ConvTranspose2d(out_dim, out_dim//2, kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn4 = nn.BatchNorm2d(out_dim//2)

        self.deconv2 = nn.ConvTranspose2d(out_dim//2, out_dim//4, kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn5 = nn.BatchNorm2d(out_dim//4)

        if double_channel:
            self.deconv3 = nn.ConvTranspose2d(out_dim//4, (out_dim//4) * 2, kernel_size=5, stride=1, padding=2)
            nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
            self.bn6 = nn.BatchNorm2d((out_dim//4) * 2)
        else:
            self.deconv3 = nn.ConvTranspose2d(out_dim//4, out_dim//4, kernel_size=5, stride=1, padding=2)
            nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
            self.bn6 = nn.BatchNorm2d(out_dim//4)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = F.relu(self.bn4(self.deconv1(x)))
        x = F.relu(self.bn5(self.deconv2(x)))
        x = F.relu(self.bn6(self.deconv3(x)))

        return x

class ConvAutoEncoder(nn.Module):

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.last_upsample = int(img_size / 8) if padding else int(img_size / 8) - 8

        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn2d4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)
        self.bn2d6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        # decoder
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.bn2d7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.bn2d8 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.bn2d9 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.bn2d10 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.bn2d11 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d12 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = imagenet_norm_batch(x)

        # encoder
        x = self.enconv1(x)
        x = F.relu(self.bn2d1(x))

        x = self.enconv2(x)
        x = F.relu(self.bn2d2(x))

        x = self.enconv3(x)
        x = F.relu(self.bn2d3(x))

        x = self.enconv4(x)
        x = F.relu(self.bn2d4(x))

        x = self.enconv5(x)
        x = F.relu(self.bn2d5(x))

        x = self.enconv6(x)
        x = F.relu(self.bn2d6(x))

        # decoder
        x = F.interpolate(x, size=(int(self.img_size / 64) - 1, int(self.img_size / 64) - 1), mode="bilinear")
        x = F.relu(self.bn2d7(self.deconv1(x)))

        x = F.interpolate(x, size=(int(self.img_size / 32), int(self.img_size / 32)), mode="bilinear")
        x = F.relu(self.bn2d8(self.deconv2(x)))

        x = F.interpolate(x, size=int(self.img_size / 16) - 1, mode="bilinear")
        x = F.relu(self.bn2d9(self.deconv3(x)))

        x = F.interpolate(x, size=int(self.img_size / 8) - 1, mode="bilinear")
        x = F.relu(self.bn2d10(self.deconv4(x)))

        x = F.interpolate(x, size=int(self.img_size / 4) - 1, mode="bilinear")
        x = F.relu(self.bn2d11(self.deconv5(x)))

        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.bn2d12(self.deconv7(x)))

        x = self.deconv8(x)
        return x

class UlsadModel(nn.Module):

    def __init__(
            self,
            image_size: tuple = (256, 256),
            lambdadir_l: float = 0.5,
            lambdadir_g: float = 0.5,
            lambdadir_a: float = 0.5,
            pad_maps: bool = False,
            backbone: str = 'wide_resnet50_2',
        ):
        super().__init__()

        self.pad_maps = pad_maps
        self.lambdadir_l = lambdadir_l
        self.lambdadir_g = lambdadir_g
        self.lambdadir_a = lambdadir_a
        self.backbone =  backbone

        #TODO: Take from config
        self.teacher_out_channels = 384
        self.image_size = image_size[0]
        self.input_size = image_size

        self.pretrain = PretrainExtractor(backbone = backbone,
                                          layers=["layer2", "layer3"],
                                          target_dim = self.teacher_out_channels)

        self.featae = FeatAE(double_channel = True)

        self.img_encoder = ConvAutoEncoder(out_channels = self.teacher_out_channels,
                                            padding = True,
                                            img_size = self.image_size)

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_global": torch.tensor(0.0),
                "qb_global": torch.tensor(0.0),
                "qa_local": torch.tensor(0.0),
                "qb_local": torch.tensor(0.0),
            }
        )

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        sigma = 4
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def compute_crossattn(self, feature_t: Tensor, feature_s: Tensor):

        n, c, h, w = feature_t.shape

        key = feature_t.reshape(n, c, -1).permute(0,2,1)
        query = feature_s.reshape(n, c, -1).permute(0,2,1)
        value = feature_t.reshape(n, c, -1).permute(0,2,1)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(c)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.permute(0,2,1).reshape(n, c, h, w)

        return attention_output

    def compute_selfattn(self, feature: Tensor):

        n, c, h, w = feature.shape

        key = feature.reshape(n, c, -1).permute(0,2,1)
        query = feature.reshape(n, c, -1).permute(0,2,1)
        value = feature.reshape(n, c, -1).permute(0,2,1)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(c)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.permute(0,2,1).reshape(n, c, h, w)

        return attention_output


    def forward(self, img):

        # img features
        with torch.no_grad():
            img_feats = self.pretrain(img)
            if self.is_set(self.mean_std):
                img_feats = (img_feats - self.mean_std["mean"]) / self.mean_std["std"]

        ae_output = self.img_encoder(img)

        # local patch feature reconstruction
        local_recons = self.featae(img_feats)

        if self.training:
            # Local loss
            distance_pl = torch.pow(img_feats - local_recons[:, : self.teacher_out_channels, :, :], 2)
            distance_pl = torch.mean(distance_pl, dim=1)
            distance_pl += self.lambdadir_l * (1 - F.cosine_similarity(img_feats, local_recons[:, : self.teacher_out_channels, :, :]))

            loss_pl = torch.mean(distance_pl)

            # Global loss
            teacher_attn = self.compute_selfattn(img_feats)
            global_attn = self.compute_crossattn(img_feats, ae_output)

            distance_pg = torch.pow(teacher_attn - global_attn, 2)
            distance_pg = torch.mean(distance_pg, dim=1)
            distance_pg += self.lambdadir_a * (1 - F.cosine_similarity(teacher_attn, global_attn))

            loss_pg = torch.mean(distance_pg)

            # Global local loss
            distance_lg = torch.pow(ae_output - local_recons[:, self.teacher_out_channels :, :, :], 2)
            distance_lg = torch.mean(distance_lg, dim=1)
            distance_lg += self.lambdadir_g * (1 - F.cosine_similarity(ae_output, local_recons[:, self.teacher_out_channels :, :, :]))

            loss_lg = torch.mean(distance_lg)

            loss = loss_pl + loss_pg + loss_lg

            return loss

        else:

            distance_pl = torch.pow(img_feats - local_recons[:, : self.teacher_out_channels, :, :], 2)
            distance_pl = torch.mean(distance_pl, dim=1)
            distance_pl += self.lambdadir_l * (1 - F.cosine_similarity(img_feats, local_recons[:, : self.teacher_out_channels, :, :]))

            map_local = torch.unsqueeze(distance_pl, dim=1)

            distance_lg = torch.pow(ae_output - local_recons[:, self.teacher_out_channels :, :, :], 2)
            distance_lg = torch.mean(distance_lg, dim=1)
            distance_lg += self.lambdadir_g * (1 - F.cosine_similarity(ae_output, local_recons[:, self.teacher_out_channels :, :, :]))

            map_global = torch.unsqueeze(distance_lg, dim=1)

            if self.pad_maps:
                map_local = F.pad(map_local, (4, 4, 4, 4))
                map_global = F.pad(map_global, (4, 4, 4, 4))
            map_local = F.interpolate(map_local, size=(self.input_size[0], self.input_size[1]), mode="bilinear")
            map_global = F.interpolate(map_global, size=(self.input_size[0], self.input_size[1]), mode="bilinear")

            if self.is_set(self.quantiles):
                map_local = 0.1 * (map_local - self.quantiles["qa_local"]) / (self.quantiles["qb_local"] - self.quantiles["qa_local"])
                map_global = 0.1 * (map_global - self.quantiles["qa_global"]) / (self.quantiles["qb_global"] - self.quantiles["qa_global"])

            map_local = self.blur(map_local)
            map_global = self.blur(map_global)

            anomaly_map_combined = 0.5 * map_local + 0.5 * map_global

            return {"anomaly_map_combined": anomaly_map_combined, "map_local": map_local, "map_global": map_global}