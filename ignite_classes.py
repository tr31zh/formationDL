import logging
from pathlib import Path

from tqdm import tqdm

from rich.console import Console
from rich.table import Table
from rich import print as pprint

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

import albumentations as A
from albumentations.pytorch import ToTensorV2


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from torchvision.models import swin_v2_b, Swin_V2_B_Weights

from torchmetrics.classification import MultilabelF1Score

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Loss, Accuracy, Precision, Recall
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import global_step_from_engine
from ignite.handlers.tqdm_logger import ProgressBar
import ignite.handlers.mlflow_logger as mlflow_logger
from ignite.handlers.param_scheduler import LRScheduler
from ignite.handlers import FastaiLRFinder

import mlflow
from mlflow.models import infer_signature
from mlflow.pytorch import log_model
from mlflow.entities import RunStatus

from livelossplot import PlotLossesIgnite

from transformers import (
    ViTForImageClassification,
    SegformerForImageClassification,
    BeitForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
    DeiTForImageClassificationWithTeacher,
    ResNetForImageClassification,
)


TMP_IGNITE_CHKPT_NAME = "best_ignite_chkpt.pt"
PT_TMP_CHK_IGNITE = Path(".").joinpath("tmp").joinpath("checkpoints")
EXP_NAME = "image_labeller"


def ensure_folder(forced_path, return_string: bool = True):
    path = forced_path.parent
    if path.is_dir() is False:
        path.mkdir(parents=True, exist_ok=True)
    return str(forced_path) if return_string is True else forced_path


def read_dataframe(path, sep=";") -> pd.DataFrame:
    try:
        return pd.read_csv(filepath_or_buffer=str(path), sep=sep)
    except:
        return None


def write_dataframe(df: pd.DataFrame, path, sep=";") -> pd.DataFrame:
    df.to_csv(path_or_buf=ensure_folder(path, return_string=True), sep=sep, index=False)
    return df


# MARK: Device
def get_device() -> str:
    """Return best available device

    Returns:
        str: device
    """
    return (
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


# MARK: Augmentations
def get_augmentation(
    train_mode: bool,
    test_mode: bool = False,
    image_size: int = 224,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.25,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> A.Compose:
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
    return A.Compose(transforms)


# Load image and convert to RGB
def load_image(file_name):
    if Path(file_name).is_file():
        path = file_name
    else:
        path = (
            Path(".")
            .joinpath("data")
            .joinpath("images")
            .joinpath(file_name)
            .with_suffix(".jpg")
        )

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# MARK: Dataset
class FldDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        train_mode: bool,
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
        if isinstance(data, str) is True:
            self.dataframe = pd.DataFrame(data={"image": [data]})
        elif isinstance(data, list) is True:
            self.dataframe = pd.DataFrame(data={"image": data})
        elif isinstance(data, pd.DataFrame) is True:
            self.dataframe = data
        else:
            raise NotImplementedError("Unknown source type")

        self.transform = get_augmentation(
            train_mode=train_mode,
            test_mode=test_mode,
            image_size=image_size,
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            mean=mean,
            std=std,
        )

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
            self.dataframe.iloc[:, 2:].values[index].astype("int8"), dtype=torch.float32
        )
        return {"image": img, "labels": labels}

    def get_image(self, index: int) -> np.ndarray:
        """Loads image at index

        Args:
            index (int): Index

        Returns:
            np.ndarray: Image
        """
        return load_image(file_name=str(self.dataframe.image.to_list()[index]))

    def get_data(self, index: int) -> list:
        """Loads labels at index

        Args:
            index (int): Index

        Returns:
            list: Labels
        """
        return self.dataframe.iloc[index]


# MARK: Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        focal_loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss

        match self.reduction:
            case "mean":
                return focal_loss.mean()
            case "sum":
                return focal_loss.sum()
            case _:
                return focal_loss


# MARK: Model
class FdlNet(nn.Module):
    def __init__(
        self,
        labels: list,
        augmentations: A.Compose,
        backbone: str = "hf_swt_t",
        device: str | None = None,
    ):
        """Creates the classification model

        Args:
            labels (list): List of target labels
            augmentations (A.Compose): Input layer augmentation
            backbone (str, optional): Selected backbone. Defaults to "hf_swt_t".
            device (str | None, optional): Selected device. If None, best device is selected. Defaults to None.
        """
        super(FdlNet, self).__init__()
        self.backbone = backbone
        self.augmentations = augmentations
        self.thresholds = None

        # Model
        chk_data = checkpoints_dict[backbone]
        self.labels = labels
        # self.encoder = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        self.encoder = chk_data["class"].from_pretrained(
            chk_data["path"],
            num_labels=len(labels),
            id2label={i: c for i, c in enumerate(labels)},
            label2id={c: i for i, c in enumerate(labels)},
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        self.linear = nn.Linear(len(labels), len(labels))
        # self.linear = nn.Linear(1000, len(labels))
        self.sigmoid = nn.Sigmoid()
        self.device = device if device is not None else get_device()

    def forward(self, x):
        x = self.encoder(x).logits
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

    def to_logger(self) -> dict:
        """Model data added to log

        Returns:
            dict: data dictionnary
        """
        return {k: getattr(self, k) for k in ["backbone", "labels", "device"]}

    def hr_desc(self):
        """Print human readable model description"""
        table = Table(title=f"Segmenter params & values")
        table.add_column("Param", justify="right", style="bold", no_wrap=True)
        table.add_column("Value")

        for k, v in self.to_logger().items():
            table.add_row(k, str(v))

        Console().print(table)

    def get_labels(self, dataset: FldDataset) -> np.array:
        """Returns dataset's ground truth data

        Args:
            dataset (FldDataset): Target dataset

        Returns:
            np.array: Ground truth
        """
        return dataset.dataframe.iloc[
            :,
            (
                2
                if len(dataset.dataframe.columns) > 1
                and dataset.dataframe.columns[1] == "site"
                else 1
            ) :,
        ].values

    def predict_propabilities(
        self,
        dataset: FldDataset,
        batch_size: int = 32,
        num_workers: int = 10,
        silent: bool = False,
    ) -> np.array:
        self.eval()
        self.to(get_device())
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        predictions = []
        for batch in tqdm(
            iterable=dataloader, desc="Predicting probabilities", disable=silent
        ):
            x, _ = batch["image"], batch["labels"]
            pred = self(x.to(get_device()))
            predictions.append(pred.cpu().detach())

        return torch.cat(predictions).numpy()

    def get_probas_and_labels(
        self,
        dataset: FldDataset,
        batch_size: int = 32,
        num_workers: int = 10,
    ):
        y_true = self.get_labels(dataset=dataset)
        y_proba = self.predict_propabilities(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        return y_proba, y_true

    def get_roc_df(self, y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return pd.DataFrame().assign(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            gmean=np.sqrt(tpr * (1 - fpr)),
        )

    def set_thresholds(
        self, dataset: FldDataset, batch_size: int = 32, num_workers: int = 10
    ) -> pd.DataFrame:
        y_proba, y_true = self.get_probas_and_labels(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.thresholds = [
            self.get_roc_df(y_true[:, i], y_proba[:, i])
            .nlargest(n=1, columns="gmean")
            .thresholds.to_list()[0]
            for i in range(len(self.labels))
        ]

        return self.thresholds

    def predict_labels(
        self, dataset: FldDataset, batch_size: int = 32, num_workers: int = 10
    ):
        y_proba = self.predict_propabilities(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        return np.stack(
            [
                np.where(y_proba[:, i] > self.thresholds[i], 1, 0)
                for i in range(len(self.labels))
            ],
            axis=-1,
        )

    def get_val_data(
        self, dataset: FldDataset, batch_size: int = 32, num_workers: int = 10
    ):
        predictions = self.predict_labels(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
        labels = self.get_labels(dataset=dataset)

        cutoff = (
            2
            if len(dataset.dataframe.columns) > 1
            and dataset.dataframe.columns[1] == "site"
            else 1
        )
        predictions_revue = dataset.dataframe.iloc[:, :cutoff]

        for i, label in enumerate(self.labels):
            predictions_revue.insert(cutoff + i * 3, f"y_{label}", predictions[:, i])
            predictions_revue.insert(cutoff + i * 3, f"ŷ_{label}", labels[:, i])
            predictions_revue.insert(
                cutoff + i * 3, f"{label}", predictions[:, i] == labels[:, i]
            )

        report_dict = classification_report(
            labels,
            predictions,
            target_names=self.labels,
            zero_division=0,
            output_dict=True,
        )
        report_data = {
            "labels": [],
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": [],
        }
        for k, v in report_dict.items():
            report_data["labels"].append(k)
            for c in ["precision", "recall", "f1-score", "support"]:
                report_data[c].append(v[c])

        return {
            "predictions_revue": predictions_revue,
            "classification_report": pd.DataFrame(data=report_data),
        }


# MARK: Train
def lcl_print(out_str: str, print_steps: str | None = "pprint"):
    match print_steps:
        case None:
            pass
        case "print":
            print(out_str)
        case "pprint":
            pprint(out_str)
        case "log":
            logging.info(out_str)


def prepare_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    print_steps: str | None = "pprint",
) -> tuple:
    lcl_print("Creating datasets", print_steps=print_steps)

    train_dataset = FldDataset(data=train, train_mode=True)
    lcl_print("Creating augmentations", print_steps=print_steps)

    val_dataset = FldDataset(data=val, train_mode=False)

    lcl_print("Creating dataloaders", print_steps=print_steps)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataset, val_dataset, train_loader, val_loader


def prepare_model(
    backbone: str,
    labels: list,
    image_size: int,
    learning_rate: float = 0.0005,
    loss_name: str = "bce",
    loss_params: dict = {"alpha": 0.5, "gamma": 1},
    device: str = get_device(),
    print_steps: str | None = "pprint",
) -> tuple:
    lcl_print("Creating model", print_steps=print_steps)
    lcl_print(f"Device used: {get_device()}", print_steps=print_steps)
    model = FdlNet(
        backbone=backbone,
        labels=labels,
        device=device,
        augmentations=get_augmentation(train_mode=False, image_size=image_size),
    ).to(device)

    lcl_print("Optimizer & criterion", print_steps=print_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def output_transform(output):
        y_pred, y = output
        return y_pred.gt(0.5).long(), y.long()

    match loss_name:
        case "bce":
            criterion = nn.BCEWithLogitsLoss()
        case "focal":
            criterion = FocalLoss(**loss_params, reduction="mean")
        case _:
            raise NotImplementedError(f"Unknown loss: {loss_name}")
    precision = Precision(
        average="macro", is_multilabel=True, output_transform=output_transform
    )
    recall = Recall(
        average="macro", is_multilabel=True, output_transform=output_transform
    )
    f1 = precision * recall * 2 / (precision + recall)
    val_metrics = {
        "Loss": Loss(criterion),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return model, criterion, optimizer, val_metrics


def prepare_trainer(
    model,
    criterion,
    optimizer,
    device,
    val_metrics,
    train_loader,
    val_loader,
    log_progress: bool = True,
    plot_loss: bool = True,
    print_steps: str | None = "pprint",
):
    lcl_print("Adding loops", print_steps=print_steps)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch["image"].to(device), batch["labels"].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch["image"].to(device), batch["labels"].to(device)
            y_pred = model(x)
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)
        metric.attach(val_evaluator, name)

    lcl_print("Adding events", print_steps=print_steps)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        metric_data = " | ".join([f"{k}: {v:.3}" for k, v in metrics.items()])
        lcl_print(
            f"Validation Results - Epoch[{trainer.state.epoch}] -> {metric_data}",
            print_steps=print_steps,
        )

    if log_progress is True:
        ProgressBar().attach(trainer, output_transform=lambda x: {"Loss": x})
    if plot_loss is True:
        callback = PlotLossesIgnite()
        callback.attach(val_evaluator)

    return trainer, train_evaluator, val_evaluator


# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return -engine.state.metrics["Loss"]


def add_checkpoint_saving(
    trainer,
    model,
    val_evaluator,
    checkpoints_n_saved: int = 1,
    score_function=score_function,
    print_steps: str | None = "pprint",
):
    lcl_print("Checkpoint saving", print_steps=print_steps)
    # Checkpoint to store n_saved best models wrt score function
    PT_TMP_CHK_IGNITE.joinpath(TMP_IGNITE_CHKPT_NAME).unlink(missing_ok=True)
    model_checkpoint = ModelCheckpoint(
        PT_TMP_CHK_IGNITE,
        n_saved=checkpoints_n_saved,
        filename_pattern=TMP_IGNITE_CHKPT_NAME,
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(
            trainer
        ),  # helps fetch the trainer's state
    )

    # Save the model after every epoch of val_evaluator is completed
    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"model": model}
    )


def add_early_stopper(
    trainer,
    val_evaluator,
    score_function=score_function,
    early_stoper_patience: int = 10,
    early_stoper_min_delta: float = 0,
    print_steps: str | None = "pprint",
):
    lcl_print("Early stopper", print_steps=print_steps)

    early_stopper = EarlyStopping(
        patience=early_stoper_patience,
        score_function=score_function,
        min_delta=early_stoper_min_delta,
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopper)


def find_lr(
    trainer, model, optimizer, train_loader, print_steps: str | None = "pprint"
):
    lcl_print("LR finder", print_steps=print_steps)
    lr_finder = FastaiLRFinder()
    try:
        # To restore the model's and optimizer's states after running the LR Finder
        with lr_finder.attach(
            trainer,
            to_save={"model": model, "optimizer": optimizer},
            start_lr=0.000000001,
            end_lr=5.0,
        ) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)
        lr_finder.apply_suggested_lr(optimizer)
        mlflow.log_params({"suggested LR": lr_finder.lr_suggestion()})
        fig, ax = plt.subplots()
        lr_finder.plot(ax=ax, skip_start=0, skip_end=0)
        mlflow.log_figure(fig, artifact_file="images/lr_finder.png")
    except Exception as e:
        lcl_print(f"failure_class: {type(e).__name__}", print_steps="print")
        lcl_print(f"failure_message: {str(e)}", print_steps="print")
    else:
        lcl_print(f"Suggested LR: {lr_finder.lr_suggestion()}", print_steps=print_steps)


def add_lr_scheduler(
    trainer, optimizer, step_size: int, gamma: float, print_steps: str | None = "pprint"
):
    lcl_print("LR scheduler", print_steps=print_steps)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        LRScheduler(StepLR(optimizer, step_size=step_size, gamma=gamma)),
    )


def log_csv(df, name, folder_name):
    write_dataframe(df=df, path=PT_TMP_CHK_IGNITE.joinpath(name + ".csv"))
    mlflow.log_artifact(PT_TMP_CHK_IGNITE.joinpath(name + ".csv"), folder_name)


def add_logger(
    trainer,
    logged_params,
    optimizer,
    train,
    train_evaluator,
    train_aug,
    val,
    val_evaluator,
    val_aug,
    model_aug,
    print_steps: str | None = "pprint",
):
    lcl_print("Creating logger", print_steps=print_steps)
    # Set experiment
    mlflow.set_experiment(EXP_NAME)

    # Define a Tensorboard logger
    mlf_logger = mlflow_logger.MLflowLogger()
    # Log experiment parameters:
    mlf_logger.log_params(logged_params)

    # Log CSVs
    log_csv(train, "train", folder_name="datasets")
    log_csv(val, "val", folder_name="datasets")

    # Log Augmentations
    for aug, name in zip([train_aug, val_aug, model_aug], ["train", "val", "model"]):
        A.save(aug, PT_TMP_CHK_IGNITE.joinpath(name + ".json"))
        mlflow.log_artifact(PT_TMP_CHK_IGNITE.joinpath(name + ".json"), "augmentations")

    # Attach the logger to the trainer to log training loss at each iteration
    mlf_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
    # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
    # of the `trainer` instead of `train_evaluator`.
    mlf_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    mlf_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    mlf_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name="lr",  # optional
    )

    # Log learning rate
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_learning_rate(engine):
        mlflow.log_metric(
            key="learning_rate",
            value=optimizer.param_groups[0]["lr"],
            step=engine.state.epoch,
        )

    return mlf_logger


def upload_model_to_logger(
    model: FdlNet,
    mlf_logger,
    val_dataset,
    device: str = get_device(),
    print_steps: str | None = "pprint",
    batch_size: int = 32,
    num_workers: int = 10,
):
    lcl_print("Uploading model", print_steps=print_steps)
    model.set_thresholds(
        dataset=val_dataset, batch_size=batch_size, num_workers=num_workers
    )

    lcl_print("Performing validation", print_steps=print_steps)
    val_data = model.get_val_data(dataset=val_dataset)
    log_csv(
        df=val_data["predictions_revue"],
        name="predictions_revue",
        folder_name="metrics",
    )
    log_csv(
        df=val_data["classification_report"],
        name="classification_report",
        folder_name="metrics",
    )
    mlflow.log_params(
        {
            f"F1_{label}": fscore
            for label, fscore in val_data["classification_report"][
                ["labels", "f1-score"]
            ].values
        }
    )

    test_batch = val_dataset[0]["image"].unsqueeze(0)
    signature = infer_signature(
        model_input=test_batch.numpy(),
        model_output=model(test_batch.to(device)).detach().cpu().numpy(),
    )
    log_model(model, "model", signature=signature)
    mlf_logger.log_params(model.to_logger())


def train_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    batch_size: int,
    max_epochs: int,
    image_size: int,
    backbone: str = "hf_swt_t",
    loss_name: str = "bce",
    loss_params: dict = {"alpha": 0.5, "gamma": 1},
    device: str = get_device(),
    checkpoints_n_saved: int = 1,
    learning_rate: float = 0.01,
    early_stoper_patience: int = 10,
    early_stoper_min_delta: float = 0,
    use_lr_finder: bool = False,
    lr_scheduler_step: int = -1,
    lr_scheduler_gamma: float = 0.9,
    print_steps: str | None = "pprint",
    log_progress: bool = True,
    plot_loss: bool = True,
    num_workers: int = 10,
) -> FdlNet:
    """Trains a multilabel classifier model

    Args:
        train_data (pd.DataFrame): Dataframe containing the training data
        val_data (pd.DataFrame): Dataframe containing the validation data
        batch_size (int): Batch size
        max_epochs (int): Max number of allowed epochs
        image_size (int): Training image size
        backbone (str, optional): Selected backbone, one of "checkpoints_dict" keys. Defaults to "hf_swt_t".
        loss_name (str, optional): Loss name, either "bce" or "focal". If "focal" is selected, loss_params may contain a value for "alpha" and "gamma". Defaults to "bce".
        loss_params (dict, optional): Arguments for the focal loss. Defaults to {"alpha": 0.5, "gamma": 1}.
        device (str, optional): Selected device, if unset, best one will be selected. Defaults to get_device().
        checkpoints_n_saved (int, optional): Number of best checkpoints to be saved. Defaults to 1.
        learning_rate (float, optional): Starting learning rate. Defaults to 0.01.
        early_stoper_patience (int, optional): Early stopper patience. If the model does not improve within patience epochs, training will stop. Defaults to 10.
        early_stoper_min_delta (float, optional): Min delta to be considered an improvement. Defaults to 0.
        use_lr_finder (bool, optional): USe LR finder to find best learning rate. HIGHLY EXPERIMENTAL. Defaults to False.
        lr_scheduler_step (int, optional): LR scheduler step. Defaults to -1.
        lr_scheduler_gamma (float, optional): LR scheduler gamma. Defaults to 0.9.
        print_steps (str | None, optional): Print steps mode. If "None", no info will be outputed. Defaults to "pprint".
        log_progress (bool, optional): Log training progress?. Defaults to True.
        plot_loss (bool, optional): Plot loss while training. Defaults to True.
        num_workers (int, optional): Num workers for paralel training. Higher values speed up training at the cost of ressources. Defaults to 10.

    Returns:
        FdlNet: Best checkpoint for the trained model.
    """
    labels = train_data.columns[2:]

    # Create datasets
    train_dataset, val_dataset, train_loader, val_loader = prepare_datasets(
        train=train_data,
        val=val_data,
        batch_size=batch_size,
        print_steps=print_steps,
        num_workers=num_workers,
    )

    # Prepare model
    model, criterion, optimizer, val_metrics = prepare_model(
        backbone=backbone,
        labels=labels,
        image_size=image_size,
        learning_rate=learning_rate,
        loss_name=loss_name,
        loss_params=loss_params,
        device=device,
        print_steps=print_steps,
    )

    # Prepare trainer
    trainer, train_evaluator, val_evaluator = prepare_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        val_metrics=val_metrics,
        train_loader=train_loader,
        val_loader=val_loader,
        log_progress=log_progress,
        plot_loss=plot_loss,
        print_steps=print_steps,
    )

    # Checkpoint saver
    add_checkpoint_saving(
        trainer=trainer,
        model=model,
        val_evaluator=val_evaluator,
        checkpoints_n_saved=checkpoints_n_saved,
        score_function=score_function,
        print_steps=print_steps,
    )

    # Add LR scheduler
    if lr_scheduler_step > 0:
        add_lr_scheduler(
            trainer=trainer,
            optimizer=optimizer,
            step_size=lr_scheduler_step,
            gamma=lr_scheduler_gamma,
            print_steps=print_steps,
        )

    # Add early stopper
    add_early_stopper(
        trainer=trainer,
        val_evaluator=val_evaluator,
        score_function=score_function,
        early_stoper_patience=early_stoper_patience,
        early_stoper_min_delta=early_stoper_min_delta,
        print_steps=print_steps,
    )

    try:
        # Add logger
        mlf_logger = add_logger(
            trainer=trainer,
            logged_params={
                "batch_size": batch_size,
                "model": model.__class__.__name__,
                "pytorch version": torch.__version__,
                "ignite version": ignite.__version__,
                "cuda version": torch.version.cuda,
                "mlflow version": mlflow.__version__,
                "dataset": "SMALL",
                "max epochs": max_epochs,
                "checkpoints_n_saved": checkpoints_n_saved,
                "device": device,
                "learning_rate": learning_rate,
                "early_stoper_patience": early_stoper_patience,
                "early_stoper_min_delta": early_stoper_min_delta,
                "lr_scheduler_step": lr_scheduler_step,
                "lr_scheduler_gamma": lr_scheduler_gamma,
                "optimizer": type(optimizer).__name__,
                "criterion": type(criterion).__name__,
                "use_lr_finder": use_lr_finder,
                "log_progress": log_progress,
                "plot_loss": plot_loss,
                "num_workers": num_workers,
            },
            optimizer=optimizer,
            train=train_data,
            train_evaluator=train_evaluator,
            train_aug=get_augmentation(train_mode=True),
            val=val_data,
            val_evaluator=val_evaluator,
            val_aug=get_augmentation(train_mode=False),
            model_aug=get_augmentation(train_mode=False),
            print_steps=print_steps,
        )

        # Log loss data only if not BCE
        if loss_params and loss_name != "bce":
            for k, v in loss_params.items():
                mlflow.log_params({f"loss_{k}": v})

        if use_lr_finder is True:
            find_lr(
                trainer=trainer,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                print_steps=print_steps,
            )

        # Train model
        lcl_print("Training model", print_steps=print_steps)
        trainer.run(train_loader, max_epochs=max_epochs)

        lcl_print("Restore best model", print_steps=print_steps)
        state_dict = torch.load(PT_TMP_CHK_IGNITE.joinpath(TMP_IGNITE_CHKPT_NAME))
        model.load_state_dict(state_dict)

        # Upload model
        upload_model_to_logger(
            model=model,
            mlf_logger=mlf_logger,
            val_dataset=val_dataset,
            device=device,
            print_steps=print_steps,
        )
    except Exception as e:
        try:
            mlflow.log_params({"failure_class": type(e).__name__})
            mlflow.log_params({"failure_message": str(e)})
        except:
            pass
        lcl_print(f"failure_class: {type(e).__name__}", print_steps=print_steps)
        lcl_print(f"failure_message: {str(e)}", print_steps=print_steps)
        if mlflow.active_run() is not None:
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))
    else:
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FINISHED))
    lcl_print("--> Trained ended", print_steps=print_steps)

    return model
