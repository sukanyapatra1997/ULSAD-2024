#----------------------------------------------------------------------------
# Created By  : Anomymous Author
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains PL Lightning Module for the ULSAD algorithm.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import torch
import tqdm

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim
from torch.utils.data import DataLoader

from anomalib.models.components import AnomalyModule

from .torch_model import UlsadModel

logger = logging.getLogger(__name__)


class Ulsad(AnomalyModule):
    """PL Lightning Module for the ULSAD algorithm.
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(hparams)

        self.config = hparams

        self.image_size = self.config.dataset.image_size
        self.model_size = self.config.model.model_size

        logger.info(f"Image size {self.image_size} Model size {self.model_size}")

        self.model = UlsadModel(image_size = self.image_size,
                                lambdadir_l = self.config.model.lambdadir_l,
                                lambdadir_g = self.config.model.lambdadir_g,
                                lambdadir_a = self.config.model.lambdadir_g,
                                backbone=self.config.model.backbone,
                                )

    @torch.no_grad()
    def pretrain_channel_mean_std(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate the mean and std of the teacher models activations.

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of channel-wise mean and std
        """
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm.tqdm(dataloader, desc="Calculate pretrained model channel mean & std", position=0, leave=True):
            y = self.model.pretrain(batch["image"].to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros((num_channels,), dtype=torch.float64, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float64, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        assert n is not None

        channel_mean = chanel_sum / n

        channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate 90% and 99.5% quantiles of the student(st) and autoencoder(ae).

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of both the 90% and 99.5% quantiles
            of both the student and autoencoder feature maps.
        """
        maps_local = []
        maps_global = []

        logger.info("Calculate Validation Dataset Quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            for img, label in zip(batch["image"], batch["label"]):
                if label == 0:  # only use good images of validation set!

                    img = img.to(self.device).unsqueeze(0)
                    outputs = self.model(img)
                    maps_local.append(outputs['map_local'])
                    maps_global.append(outputs['map_global'])

        qa_global, qb_global = self._get_quantiles_of_maps(maps_global)
        qa_local, qb_local = self._get_quantiles_of_maps(maps_local)

        return {"qa_global": qa_global, "qb_global": qb_global, "qa_local": qa_local, "qb_local": qb_local}

    def _get_min_max(self, maps: list[Tensor]) -> tuple[Tensor, Tensor]:

        logger.info(f"Computing min max")

        maps_flat = self.reduce_tensor_elems(torch.cat(maps))

        qa = torch.min(maps_flat).to(self.device)
        qb = torch.max(maps_flat).to(self.device)
        return qa, qb

    def _get_quantiles_of_maps(self, maps: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Calculate 90% and 99.5% quantiles of the given anomaly maps.

        If the total number of elements in the given maps is larger than 16777216
        the returned quantiles are computed on a random subset of the given
        elements.

        Args:
            maps (list[Tensor]): List of anomaly maps.

        Returns:
            tuple[Tensor, Tensor]: Two scalars - the 90% and the 99.5% quantile.
        """
        logger.info(f"qa: {self.config.model.qa} qb: {self.config.model.qb}")

        maps_flat = self.reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=self.config.model.qa).to(self.device)
        qb = torch.quantile(maps_flat, q=self.config.model.qb).to(self.device)
        return qa, qb

    def reduce_tensor_elems(self, tensor: torch.Tensor, m=2**24) -> torch.Tensor:
        """Flattens n-dimensional tensors,  selects m elements from it
        and returns the selected elements as tensor. It is used to select
        at most 2**24 for torch.quantile operation, as it is the maximum
        supported number of elements.
        https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

        Args:
            tensor (torch.Tensor): input tensor from which elements are selected
            m (int): number of maximum tensor elements. Default: 2**24

        Returns:
                Tensor: reduced tensor
        """
        tensor = torch.flatten(tensor)

        if len(tensor) > m:
            # select a random subset with m elements.
            perm = torch.randperm(len(tensor), device=tensor.device)
            idx = perm[:m]
            tensor = tensor[idx]
        return tensor

    def configure_optimizers(self) -> optim.Optimizer:

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.model.lr,
                                     weight_decay=self.config.model.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.config.model.lr_milestones,
                                                         gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self) -> None:
        """Calculate or load the channel-wise mean and std of the training dataset and push to the model."""
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std =  self.pretrain_channel_mean_std(self.trainer.datamodule.train_dataloader())
            self.model.mean_std.update(channel_mean_std)

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Training step for EfficientAd returns the student, autoencoder and combined loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Loss.
        """
        del args, kwargs  # These variables are not used.

        loss = self.model(batch["image"])

        return {"loss": loss}

    def on_validation_start(self) -> None:
        """
        Calculate the feature map quantiles of the validation dataset and push to the model.
        """
        if (self.current_epoch + 1) == self.trainer.max_epochs:
            map_norm_quantiles = self.map_norm_quantiles(self.trainer.datamodule.val_dataloader())
            self.model.quantiles.update(map_norm_quantiles)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of EfficientAd returns anomaly maps for the input image batch

        Args:
          batch (dict[str, str | Tensor]): Input batch

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])["anomaly_map_combined"]

        return batch


class UlsadLightning(Ulsad):
    """PL Lightning Module for the EfficientAd Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(hparams)
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
