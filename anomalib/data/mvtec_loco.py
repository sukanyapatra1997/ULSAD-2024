from __future__ import annotations

import logging
import cv2
import numpy as np
from torch import Tensor
from pathlib import Path
from typing import Sequence

import albumentations as A
from pandas import DataFrame
from overrides import override

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
    masks_to_boxes,
    read_image
)
logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")


CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


def make_mvtecloco_dataset(
    root: str | Path, split: str | Split | None = None, extensions: Sequence[str] | None = None
) -> DataFrame:

    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = Path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if (f.suffix in extensions) and (f.parts[-3] in ["train", "test", "validation"])]

    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # samples = samples.loc[samples.label != 'structural_anomalies']
    # samples = samples.loc[samples.label != 'logical_anomalies']

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    # samples.loc[
    #     (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL), "mask_path"
    # ] = mask_samples.image_path.values

    # assert that the right mask files are associated with the right test images
    # if len(samples.loc[samples.label_index == LabelName.ABNORMAL]):
    #     assert (
    #         samples.loc[samples.label_index == LabelName.ABNORMAL]
    #         .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
    #         .all()
    #     ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files in 'ground_truth' \
    #             folder follow the same naming convention as the anomalous images in the dataset (e.g. image: \
    #             '000.png', mask: '000.png' or '000_mask.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecLocoDataset(AnomalibDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        category: str,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self) -> None:
        self.samples = make_mvtecloco_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)

    # @override
    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[dict[str, Tensor], dict[str, str | Tensor]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """

        image_path = self._samples.iloc[index].image_path
        mask_path = self._samples.iloc[index].mask_path
        label_index = self._samples.iloc[index].label_index

        image = read_image(image_path)
        item = dict(image_path=image_path, label=label_index)

        if self.task == TaskType.CLASSIFICATION:
            transformed = self.transform(image=image)
            item["image"] = transformed["image"]
        elif self.task in (TaskType.DETECTION, TaskType.SEGMENTATION):
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.

            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = np.zeros(shape=image.shape[:2])
                # for i in range(3):
                #     maskpath = Path((str(image_path)[:-4]).replace("test", "ground_truth") + "/00{}.png".format(i))
                #     if Path.exists(maskpath):
                #         mask += cv2.imread(maskpath.as_posix(), flags=0)
                mask_dir = Path((str(image_path)[:-4]).replace("test", "ground_truth"))
                maskpaths = [x.name for x in mask_dir.iterdir() if x.is_file()]
                for fn in maskpaths:
                    mask += cv2.imread(mask_dir.joinpath(fn).as_posix(), flags=0)
                mask[mask > 0] = 255

                mask /=  255.0

            transformed = self.transform(image=image, mask=mask)

            item["image"] = transformed["image"]
            item["mask_path"] = mask_path
            item["mask"] = transformed["mask"]

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return item


class MVTecLoco(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = MVTecLocoDataset(
            task=task, transform=transform_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = MVTecLocoDataset(
            task=task, transform=transform_eval, split=Split.TEST, root=root, category=category
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            raise NotImplementedError
            # download_and_extract(self.root, DOWNLOAD_INFO)

