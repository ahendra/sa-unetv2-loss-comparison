import os
import sys
import time
from pathlib import Path
from typing import Callable, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras.optimizers import Adam

from config import DriveConfig, StareConfig, WEIGHTS_DIR
from src.models import build_sa_unetv2


class _ProgressCallback(keras.callbacks.Callback):
    """Prints epoch-level progress with key metrics."""

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total = total_epochs
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n  Training started — {self.total} max epochs")
        print(f"  {'Epoch':>6}  {'Loss':>10}  {'Val Loss':>10}  {'Val Acc':>9}  {'Elapsed':>10}")
        print("  " + "-" * 55)

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        loss = logs.get('loss', float('nan'))
        val_loss = logs.get('val_loss', float('nan'))
        val_acc = logs.get('val_accuracy', float('nan'))
        print(
            f"  {epoch + 1:>6}/{self.total}"
            f"  {loss:>10.5f}"
            f"  {val_loss:>10.5f}"
            f"  {val_acc:>9.4f}"
            f"  {elapsed:>8.0f}s",
            flush=True,
        )

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        print(f"\n  Training finished in {elapsed:.0f}s")


class ModelTrainer:
    """Handles model creation, compilation, and training for one loss function."""

    def __init__(self, cfg: Union[DriveConfig, StareConfig]):
        self.cfg = cfg

    def train(
        self,
        loss_name: str,
        loss_fn: Callable,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> keras.Model:
        weight_path = self._weight_path(loss_name)
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        model = build_sa_unetv2(
            input_size=self.cfg.input_size,
            start_neurons=self.cfg.start_neurons,
            block_size=self.cfg.block_size,
            rate=self.cfg.drop_rate,
        )
        model.compile(
            optimizer=Adam(learning_rate=self.cfg.learning_rate),
            loss=loss_fn,
            metrics=['accuracy'],
        )

        print(f"\n  Model: SA-UNetV2 | Loss: {loss_name} | Dataset: {self.cfg.name}")
        print(f"  Train samples: {len(x_train)}  Val samples: {len(x_val)}")
        print(f"  Weights will be saved to: {weight_path}")

        callbacks = [
            _ProgressCallback(self.cfg.epochs),
            ModelCheckpoint(
                str(weight_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.cfg.reduce_lr_patience,
                min_lr=self.cfg.reduce_lr_min,
                verbose=1,
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        model.fit(
            x_train, y_train,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        return model

    def load_weights(self, loss_name: str) -> keras.Model:
        """Build model and load saved best weights for a given loss."""
        weight_path = self._weight_path(loss_name)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"No saved weights found for '{loss_name}' at {weight_path}"
            )
        model = build_sa_unetv2(
            input_size=self.cfg.input_size,
            start_neurons=self.cfg.start_neurons,
            block_size=self.cfg.block_size,
            rate=self.cfg.drop_rate,
        )
        model.load_weights(str(weight_path))
        return model

    def weights_exist(self, loss_name: str) -> bool:
        return self._weight_path(loss_name).exists()

    def _weight_path(self, loss_name: str) -> Path:
        dataset_dir = self.cfg.name.lower()
        return WEIGHTS_DIR / dataset_dir / f"{self.cfg.name.lower()}_{loss_name}.weights.h5"
