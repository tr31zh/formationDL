from pathlib import Path
from datetime import datetime as dt

from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelF1Score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import (
    ViTForImageClassification,
    SegformerForImageClassification,
    BeitForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
    DeiTForImageClassificationWithTeacher,
    ResNetForImageClassification,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    RichProgressBar,
    DeviceStatsMonitor,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner

# Returns the best device available
g_device = (
    "mps"
    if torch.backends.mps.is_built() is True
    else "cuda" if torch.backends.cuda.is_built() else "cpu"
)

checkpoints_dict = {
    "hf_vit_g16": {
        "path": "google/vit-base-patch16-224-in21k",
        "name": "Google ViT 16",
        "link": "https://huggingface.co/google/vit-base-patch16-224-in21k",
        "class": ViTForImageClassification,
    },
    "hf_bb_16": {
        "path": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "name": "BEiT (base-sized model, fine-tuned on ImageNet-22k)",
        "link": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k",
        "class": BeitForImageClassification,
    },
    "hf_seg": {
        "path": "nvidia/mit-b0",
        "name": "Segformer",
        "link": "https://huggingface.co/nvidia/mit-b0",
        "class": SegformerForImageClassification,
    },
    "hf_bl_16": {
        "path": "microsoft/beit-large-patch16-224-pt22k-ft22k",
        "name": "BEiT (large-sized model, fine-tuned on ImageNet-22k)",
        "link": "https://huggingface.co/microsoft/beit-large-patch16-224-pt22k-ft22k",
        "class": BeitForImageClassification,
    },
    "hf_vit_g32": {
        "path": "google/vit-large-patch32-384",
        "name": "Vision Transformer (large-sized model)",
        "link": "https://huggingface.co/google/vit-large-patch32516-384",
        "class": ViTForImageClassification,
    },
    "hf_swt_t": {
        "path": "microsoft/swin-tiny-patch4-window7-224",
        "name": "Swin Transformer (tiny-sized model)",
        "link": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224",
        "class": SwinForImageClassification,
    },
    "hf_cnx_t": {
        "path": "facebook/convnext-tiny-224",
        "name": "ConvNeXT (tiny-sized model)",
        "link": "https://huggingface.co/facebook/convnext-tiny-224",
        "class": ConvNextForImageClassification,
    },
    "hf_det_b": {
        "path": "facebook/deit-base-distilled-patch16-224",
        "name": "Distilled Data-efficient Image Transformer (base-sized model)",
        "link": "https://huggingface.co/facebook/deit-base-distilled-patch16-224",
        "class": DeiTForImageClassificationWithTeacher,
    },
    "hf_swt_l": {
        "path": "microsoft/swin-large-patch4-window12-384-in22k",
        "name": "Swin Transformer (large-sized model)",
        "link": "https://huggingface.co/microsoft/swin-large-patch4-window12-384-in22k",
        "class": SwinForImageClassification,
    },
    "hf_deit_s": {
        "path": "facebook/deit-small-patch16-224",
        "name": "Data-efficient Image Transformer (small-sized model)",
        "link": "https://huggingface.co/facebook/deit-small-patch16-224",
        "class": ViTForImageClassification,
    },
    "hf_seg_b3": {
        "path": "nvidia/mit-b3",
        "name": "SegFormer (b3-sized) encoder pre-trained-only",
        "link": "https://huggingface.co/nvidia/mit-b3",
        "class": SegformerForImageClassification,
    },
    "hf_vit_gl": {
        "path": "google/vit-large-patch16-224",
        "name": "Vision Transformer (large-sized model)",
        "link": "https://huggingface.co/google/vit-large-patch16-224",
        "class": ViTForImageClassification,
    },
    "hf_resnet": {
        "path": "microsoft/resnet-50",
        "name": "ResNet-50 v1.5",
        "link": "https://huggingface.co/microsoft/resnet-50",
        "class": ResNetForImageClassification,
    },
}


# Load image and convert to RGB
def load_image(file_name):
    path = (
        Path(".")
        .joinpath("data")
        .joinpath("images")
        .joinpath(file_name)
        .with_suffix(".jpg")
    )

    try:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(file_name)
    return img


class FldDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        train_mode: bool,
        columns: list = [],
        image_size: int = 224,
        brightness_limit: float = 0.15,
        contrast_limit: float = 0.25,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        test_mode: bool = False,
    ) -> None:
        """Main dataset class, yields an item composed of an image and its attached labels

        Args:
            dataframe (pd.DataFrame): Source dataframe
            train_mode (bool): If true, will add training augmentations to transform
            columns (list, optional): List of label columns, if none, all labels will be selected. Defaults to [].
            image_size (int, optional): Image size. Defaults to 224.
            brightness_limit (float, optional): Max brightness augmentation. Defaults to 0.15.
            contrast_limit (float, optional): Max contrast augmentation. Defaults to 0.25.
            mean (tuple, optional): Mean for normalization. Defaults to (0.485, 0.456, 0.406).
            std (tuple, optional): Standard deviation for normalization. Defaults to (0.229, 0.224, 0.225).
            test_mode (bool, optional): If true images will not be normalized and transformed into tensors. Defaults to False.
        """
        super().__init__()
        self.dataframe = dataframe
        if columns:
            self.dataframe = self.dataframe[["image"] + columns]
        transforms = [
            A.LongestMaxSize(max_size=image_size, p=1),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, p=1),
        ]
        if train_mode is True:
            transforms.extend(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.3),
                ]
            )
        if test_mode is False:
            transforms.extend([A.Normalize(mean=mean, std=std, p=1), ToTensorV2()])
        self.transform = A.Compose(transforms)

    def __len__(self) -> int:
        """Lenght of the dataset

        Returns:
            int: Dataset length
        """
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> dict:
        """Returns the item at selected index

        Args:
            index (int): Index

        Returns:
            dict: Dictionnary containg image aand labels
        """
        img = self.transform(image=self.get_image(index=index))["image"]
        if self.dataframe.shape[1] == 1:
            return {"image": img, "labels": torch.tensor(0, dtype=torch.float32)}
        labels = torch.tensor(
            self.dataframe.iloc[:, 1:].values[index].astype("int8"), dtype=torch.float32
        )
        return {"image": img, "labels": labels}

    def get_image(self, index: int) -> np.ndarray:
        """Loads image at index

        Args:
            index (int): Index

        Returns:
            np.ndarray: Image
        """
        return load_image(file_name=str(self.dataframe.image.to_list()[index]) + ".jpg")

    def get_data(self, index: int) -> list:
        """Loads labels at index

        Args:
            index (int): Index

        Returns:
            list: Labels
        """
        return self.dataframe.iloc[index]


class FdlNet(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        max_epochs: int,
        num_workers,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        backbone: str = "hf_swt_t",
    ) -> None:
        """Lightning model class

        Args:
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            max_epochs (int): Max training epochs
            num_workers (int): Worker count
            train (pd.DataFrame): Train data
            val (pd.DataFrame): Validation data
            test (pd.DataFrame): Test data
            backbone (str, optional): Selected backbone. Defaults to "hf_swt_t".
        """
        super().__init__()

        self.model_name = backbone.replace("_", "")

        # Encoder
        self.backbone = backbone

        # Model
        chk_data = checkpoints_dict[backbone]
        labels = train.columns[1:]
        self.labels = labels
        self.encoder = chk_data["class"].from_pretrained(
            chk_data["path"],
            num_labels=len(labels),
            id2label={i: c for i, c in enumerate(labels)},
            label2id={c: i for i, c in enumerate(labels)},
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )

        self.flatten = nn.Flatten()
        self.linear_out = nn.LazyLinear(len(labels))

        self.sigmoid = nn.Sigmoid()

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.starting_lr = self.learning_rate
        self.num_workers = num_workers
        self.max_epochs = max_epochs

        # Loss
        self.criterion = nn.BCELoss()

        # Validation metrics
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MultilabelF1Score(num_labels=len(labels), average="macro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MultilabelF1Score(num_labels=len(labels), average="macro")

        # self.train_acc = MultilabelAccuracy(num_labels=len(labels), average="weighted")
        # self.val_acc = MultilabelAccuracy(num_labels=len(labels), average="weighted")

        # dataframes
        self.train_data = train
        self.val_data = val
        self.test_data = test

        self._thresholds = None
        self._thresholds_source = None

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        x = self.encoder(x)
        if self.backbone in checkpoints_dict.keys():
            x = x.logits
        x = self.flatten(x)
        x = self.linear_out(x)
        x = self.sigmoid(x)
        return x

    def get_dataloader(self, dataset_name: str) -> DataLoader:
        """Returns selected dataloader

        Args:
            dataset_name (str): Dataloader type

        Returns:
            DataLoader: Dataloader
        """
        match dataset_name:
            case "train":
                return DataLoader(
                    dataset=FldDataset(dataframe=self.train_data, train_mode=True),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            case "val":
                return DataLoader(
                    dataset=FldDataset(dataframe=self.val_data, train_mode=False),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            case "test":
                return DataLoader(
                    dataset=FldDataset(dataframe=self.test_data, train_mode=False),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

    def get_probas_and_labels(self, y_true=None, y_proba=None, dataset: str = "val"):
        if y_true is None:
            y_true = self.get_labels(dataset=dataset)
        if y_proba is None:
            y_proba = self.predict_propabilities()

        return y_proba, y_true

    def get_roc_df(self, y_true, y_proba, dataset: str = "val"):
        y_proba, y_true = self.get_probas_and_labels(y_true, y_proba, dataset)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return pd.DataFrame().assign(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            gmean=np.sqrt(tpr * (1 - fpr)),
        )

    def get_thresholds(
        self,
        y_true=None,
        y_proba=None,
        dataset="val",
        return_dict: bool = False,
    ):
        y_proba, y_true = self.get_probas_and_labels(y_true, y_proba, dataset)

        thresholds = [
            self.get_roc_df(y_true[:, i], y_proba[:, i])
            .nlargest(n=1, columns="gmean")
            .thresholds.to_list()[0]
            for i in range(len(self.labels))
        ]
        if return_dict is True:
            return {l: t for l, t in zip(self.labels, thresholds)}
        else:
            return thresholds

    def predict_propabilities(self):
        self.eval()
        self.to(g_device)
        dataloader = self.val_dataloader()
        predictions = []
        for batch in tqdm(iterable=dataloader, desc="Predicting probabilities"):
            x, _ = batch["image"], batch["labels"]
            pred = self(x.to(g_device))
            predictions.append(pred.cpu().detach())

        return torch.cat(predictions).numpy()

    def get_labels(self, dataset: str = "val"):
        return (
            torch.stack(
                [sample["labels"].int() for sample in self.val_dataloader().dataset]
            )
            .detach()
            .cpu()
            .numpy()
        )

    def predict_labels(
        self,
        y_true=None,
        y_proba=None,
        dataset="val",
        thresholds=None,
        disable_progress=False,
    ):
        if y_proba is None:
            self.predict_propabilities()
        if thresholds is None:
            thresholds = self.get_thresholds(
                y_true=y_true, y_proba=y_proba, dataset=dataset
            )
        if isinstance(thresholds, dict):
            thresholds = [thresholds[var] for var in self.labels]

        return np.stack(
            [
                np.where(y_proba[:, i] > thresholds[i], 1, 0)
                for i in range(len(self.labels))
            ],
            axis=-1,
        )

    def predict_to_dataframe(self, dataset: pd.DataFrame):
        thresholds = self.get_thresholds()
        probas = self.predict_propabilities()
        labels = np.stack(
            [
                np.where(probas[:, i] > thresholds[i], 1, 0)
                for i in range(len(self.labels))
            ],
            axis=-1,
        )
        for i, label in enumerate(self.labels):
            insert_position = dataset.columns.to_list().index(label) + 1
            dataset = dataset.drop([label, f"p_{label}"], axis=1, errors="ignore")
            dataset.insert(insert_position, label, labels[:, i])
            dataset.insert(insert_position, f"p_{label}", probas[:, i])
        return dataset

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def compute_loss(self, preds, targets):
        return self.criterion(preds, targets)

    def training_step(self, batch, batch_idx):
        x, target = batch["image"], batch["labels"]
        x = self(x)
        loss = self.compute_loss(x, target)
        self.log_dict({"train/loss": loss.item()})

        self.mean_train_loss(loss, weight=target.shape[0])
        self.mean_train_f1(x, target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, target = batch["image"], batch["labels"]
        x = self(x)
        loss = self.compute_loss(x, target)
        self.log_dict({"val/loss": loss.item()})

        self.mean_valid_loss.update(loss, weight=target.shape[0])
        self.mean_valid_f1.update(x, target)

        return loss

    def on_validation_epoch_end(self):
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, target = batch["image"], batch["labels"]
        x = self(x)
        loss = self.compute_loss(x, target)
        self.log_dict({"test_loss": loss.item()})
        return loss

    def thresholds(self, dataset="val"):
        if self._thresholds is None or dataset != self._thresholds_source:
            self._thresholds = self.get_thresholds(dataset=dataset)
            self._thresholds_source = dataset
        return self._thresholds

    def classification_report_as_dict(
        self,
        dataset: str = "val",
        y_proba=None,
        y_true=None,
        y_pred=None,
        disable_progress: bool = False,
    ):
        y_proba, y_true = self.get_probas_and_labels(
            y_true=y_true,
            y_proba=y_proba,
            dataset=dataset,
            disable_progress=disable_progress,
        )
        if y_pred is None:
            y_pred = self.predict_labels(y_true=y_true, y_proba=y_proba)

        if len(self.labels) > 1:
            return classification_report(
                y_true,
                y_pred,
                target_names=self.labels,
                zero_division=0,
                output_dict=True,
            )
        else:
            return {"F1 score": f1_score(y_true=y_true, y_pred=y_pred)}


def get_trainer(model, checkpoints_path, patience=10, log_every_n_steps=5):
    callbacks = [
        RichProgressBar(),
        EarlyStopping(
            monitor="val/loss",
            patience=patience,
            min_delta=0.0005,
        ),
        DeviceStatsMonitor(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    callbacks.append(
        ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
            auto_insert_metric_name=False,
            filename=model.model_name
            + "-val_loss={val/loss:.3f}-val_f1={val/f1:.2f}-{epoch}-train_loss={train/loss:.3f}-train_f1={train/f1:.3f}-{step}",
        )
    )
    return Trainer(
        default_root_dir=str(checkpoints_path),
        # accelerator="cpu",
        max_epochs=model.max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
    )


def tune_trainer(
    model,
    trainer: Trainer,
    tune_options: list = ["find_lr", "find_bs"],
    find_bs_mode: str = "binsearch",
):
    tuner = Tuner(trainer=trainer)
    if "find_lr" in tune_options:
        tuner.lr_find(model)
    if "find_bs" in tune_options:
        tuner.scale_batch_size(model=model, mode=find_bs_mode)
