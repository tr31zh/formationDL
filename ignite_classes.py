import logging
from pathlib import Path

from tqdm import tqdm

from rich.console import Console
from rich.table import Table
from rich import print as pprint

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from torchmetrics.classification import MultilabelF1Score

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Loss
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
from matplotlib.figure import Figure

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
    return "cpu"
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


# MARK: Model
class FdlNet(nn.Module):

    def __init__(
        self,
        labels,
        augmentations: A.Compose,
        backbone: str = "hf_swt_t",
        device: str | None = None,
    ):
        """Create model

        Args:
            architecture (str): Base architecture, see https://github.com/qubvel/segmentation_models for list of architectures
            encoder_name (str): Encoder, see https://github.com/qubvel/segmentation_models for available encoders
            transform (A.Compose): Transformation
            device (str, optional): Selected device. Defaults to None.
        """
        super(FdlNet, self).__init__()
        self.backbone = backbone
        self.augmentations = augmentations
        self.thresholds = None

        # Model
        chk_data = checkpoints_dict[backbone]
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
        self.device = device if device is not None else get_device()

    def forward(self, x):
        x = self.encoder(x)
        if self.backbone in checkpoints_dict.keys():
            x = x.logits
        x = self.flatten(x)
        x = self.linear_out(x)
        x = self.sigmoid(x)
        return x

    def to_logger(self) -> dict:
        return {k: getattr(self, k) for k in ["backbone", "labels", "device"]}

    def hr_desc(self):
        """Print human readable model description"""
        table = Table(title=f"Segmenter params & values")
        table.add_column("Param", justify="right", style="bold", no_wrap=True)
        table.add_column("Value")

        for k, v in self.to_logger().items():
            table.add_row(k, str(v))

        Console().print(table)

    def get_labels(self, dataset: FldDataset):
        return (
            torch.stack([sample["labels"].int() for sample in dataset])
            .detach()
            .cpu()
            .numpy()
        )

    def predict_propabilities(
        self,
        dataset: FldDataset,
        batch_size: int = 32,
        num_workers: int = 10,
        silent: bool = False,
    ):
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

    def predict_to_dataframe(
        self, dataset: FldDataset, batch_size: int = 32, num_workers: int = 10
    ):
        labels = self.predict_labels(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
        probas = self.predict_propabilities(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
        for i, label in enumerate(self.labels):
            insert_position = dataset.columns.to_list().index(label) + 1
            dataset = dataset.drop([label, f"p_{label}"], axis=1, errors="ignore")
            dataset.insert(insert_position, label, labels[:, i])
            dataset.insert(insert_position, f"p_{label}", probas[:, i])
        return dataset

    def classification_report_as_dict(
        self, dataset: FldDataset, batch_size: int = 32, num_workers: int = 10
    ):
        y_proba, y_true = self.get_probas_and_labels(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        y_pred = self.predict_labels(
            dataset=dataset,
            y_true=y_true,
            y_proba=y_proba,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return classification_report(
            y_true,
            y_pred,
            target_names=self.labels,
            zero_division=0,
            output_dict=True,
        )


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
    print_steps: str | None = "pprint",
) -> tuple:
    lcl_print("Creating datasets", print_steps=print_steps)

    train_dataset = FldDataset(data=train, train_mode=True)
    lcl_print("Creating augmentations", print_steps=print_steps)

    val_dataset = FldDataset(data=val, train_mode=False)

    lcl_print("Creating dataloaders", print_steps=print_steps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


def prepare_model(
    backbone: str,
    labels: list,
    image_size: int,
    learning_rate: float = 0.0005,
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
    criterion = nn.BCEWithLogitsLoss()
    val_metrics = {
        "loss": Loss(criterion),
        "F1": MultilabelF1Score(num_labels=len(labels), average="macro"),
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
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
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
        lcl_print(
            f"Validation Results - Epoch[{trainer.state.epoch}] -> Dice: {metrics['Dice']:.5f} IoU: {metrics['IoU']:.5f}",
            print_steps=print_steps,
        )

    if log_progress is True:
        ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x})
    if plot_loss is True:
        callback = PlotLossesIgnite()
        callback.attach(val_evaluator)

    return trainer, train_evaluator, val_evaluator


# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["loss"]


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
    try:
        lr_finder = FastaiLRFinder()

        # To restore the model's and optimizer's states after running the LR Finder
        with lr_finder.attach(
            trainer,
            to_save={"model": model, "optimizer": optimizer},
            start_lr=0.00001,
            end_lr=1.0,
        ) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)

        lr_finder.apply_suggested_lr(optimizer)
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
    val_full,
    val_dataset,
    device: str = get_device(),
    log_progress: bool = True,
    print_steps: str | None = "pprint",
    batch_size: int = 32,
    num_workers: int = 10,
):
    lcl_print("Uploading model", print_steps=print_steps)
    state_dict = torch.load(PT_TMP_CHK_IGNITE.joinpath(TMP_IGNITE_CHKPT_NAME))
    model.load_state_dict(state_dict)

    model.set_thresholds(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        silent=log_progress is False,
    )

    log_csv(val_full, "val_full", folder_name="datasets")
    lcl_print("Performing validation", print_steps=print_steps)
    eval_data = model.predict_to_dataframe(
        dataset=val_dataset, batch_size=batch_size, num_workers=num_workers
    )

    lcl_print("Uploading datasets", print_steps=print_steps)
    log_csv(df=eval_data, name="eval_data", folder_name="metrics")

    mlflow.log_figure(
        model.demo_plant(df=val_full, treatment="DC", sample_count=6),
        artifact_file="images/plant_demo.png",
    )

    test_batch, _ = val_dataset[0]
    test_batch = test_batch.unsqueeze(0)
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
    upload_model: bool = True,
    is_test: bool = False,
    skip_logger: bool = False,
    num_workers: int = 10,
) -> FdlNet:
    labels = train_data.columns[:2]

    # Create datasets
    train_dataset, val_dataset, train_loader, val_loader = prepare_datasets(
        train=train_data,
        val=val_data,
        batch_size=batch_size,
        print_steps=print_steps,
    )

    # Prepare model
    model, criterion, optimizer, val_metrics = prepare_model(
        backbone=backbone,
        labels=labels,
        image_size=image_size,
        learning_rate=learning_rate,
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

    if use_lr_finder is True:
        find_lr(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
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

    if skip_logger is True:
        lcl_print("Training model", print_steps=print_steps)
        trainer.run(train_loader, max_epochs=max_epochs)
        return model

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
                "encoder_name": encoder_name,
                "test_run": is_test,
                "use_lr_finder": use_lr_finder,
                "log_progress": log_progress,
                "plot_loss": plot_loss,
                "upload_model": upload_model,
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

        # Train model
        lcl_print("Training model", print_steps=print_steps)
        trainer.run(train_loader, max_epochs=max_epochs)

        # Upload model
        if upload_model is True:
            upload_model_to_logger(
                model=model,
                mlf_logger=mlf_logger,
                val_full=val_data,
                val_dataset=val_dataset,
                device=device,
                log_progress=log_progress,
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
