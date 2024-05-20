import torch
import random
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import Tensor
from ar_model import ARModel
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union


class NMPDataset(Dataset):

    def __init__(self,
                 fold: str,
                 audio_dir: str or Path,
                 meta_path: str or Path,
                 sample_rate: int,
                 packet_dim: int,
                 extra_pred_dim: int,
                 nn_context_dim: int,
                 ar_context_dim: int,
                 ar_order: int,
                 ar_fade_dim: int,
                 diagonal_load: float,
                 steps_per_epoch: Union[int, None] = None,
                 batch_size: Union[int, None] = None,
                 ):

        self.fold = fold
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.ar_order = ar_order
        self.ar_fade_dim = ar_fade_dim
        self.diagonal_load = diagonal_load
        self.packet_dim = packet_dim
        self.pred_dim = packet_dim + extra_pred_dim
        self.nn_context_dim = nn_context_dim
        self.ar_context_dim = ar_context_dim
        self.output_dim = (nn_context_dim + 1) * packet_dim + extra_pred_dim
        self.chunk_dim = self.output_dim + ar_context_dim * packet_dim
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

        self.ar_model = ARModel(ar_order, diagonal_load)
        self.up_ramp = np.linspace(0, 1, self.ar_fade_dim)
        self.down_ramp = np.linspace(1, 0, self.ar_fade_dim)

        meta = pd.read_csv(meta_path)
        self.meta = meta[meta['subset'] == fold]
        self.num_audio_files = len(self.meta)

    def __len__(self):
        if self.fold == 'training':
            return self.steps_per_epoch * self.batch_size
        else:
            return self.num_audio_files

    def __getitem__(self, index):
        # Randomize index at training time
        if self.fold == 'training':
            index = random.randint(0, self.num_audio_files - 1)

        # Read metadata
        row = self.meta.iloc[index]

        # Medley-solos-DB file path; modify this line of code if another dataset is used
        filepath = Path(self.audio_dir, f"Medley-solos-DB_{self.fold}-{row['instrument_id']}_{row['uuid4']}.wav")

        # Load audio file
        wav, __ = librosa.load(filepath, sr=self.sample_rate, mono=True)

        if self.fold == 'training':
            # Data augmentation is not implemented yet
            wav = self._augment(wav)
            # Randomize the training chunk within the audio file
            idx = random.randint(0, len(wav) - self.chunk_dim - 1)
            chunk = wav[idx:idx + self.chunk_dim]
        else:
            # Use the very first chunk in every audio file for validation and test
            chunk = wav[:self.chunk_dim]

        # AR model contribution
        ar_data = self._get_ar_data(chunk)

        # Ground-truth audio data
        true = chunk[None, -self.output_dim:]

        # Valid neural network input, obtained by zeroing out the samples to be predicted
        past = true.copy()
        past[:, -self.pred_dim:] = 0.

        return Tensor(true), Tensor(past), Tensor(ar_data)

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        # TODO: left for future work.
        return wav

    def _get_ar_data(self, chunk: np.ndarray) -> np.ndarray:
        ar_data = np.zeros(self.output_dim)

        for i in range(self.nn_context_dim + 1):
            idx = i * self.packet_dim
            valid = chunk[idx: idx + self.ar_context_dim * self.packet_dim]
            steps = self.pred_dim if i == self.nn_context_dim else self.packet_dim + self.ar_fade_dim
            ar_pred = self.ar_model.predict(valid=valid, steps=steps)
            ar_pred = self._apply_ar_fade(frame=ar_pred, index=i)
            ar_data[idx: idx + len(ar_pred)] += ar_pred

        return ar_data[None, :]

    def _apply_ar_fade(self, frame: np.ndarray, index: int) -> np.ndarray:
        if self.ar_fade_dim:
            # Fade-out
            if index == 0:
                frame[-self.ar_fade_dim:] *= self.down_ramp
            # Fade-in
            elif index == self.nn_context_dim:
                frame[:self.ar_fade_dim] *= self.up_ramp
            # Cross-fade
            else:
                frame[:self.ar_fade_dim] *= self.up_ramp
                frame[-self.ar_fade_dim:] *= self.down_ramp

        return frame


class NMPDataModule(pl.LightningDataModule):
    def __init__(self,
                 audio_dir: str,
                 meta_path: str,
                 sample_rate: int,
                 packet_dim: int,
                 extra_pred_dim: int,
                 nn_context_dim: int,
                 ar_context_dim: int,
                 ar_order: int,
                 ar_fade_dim: int,
                 diagonal_load: float,
                 steps_per_epoch: int,
                 batch_size: int,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.kwargs = {'num_workers': 8, 'persistent_workers': True}

        self.train_dataset = NMPDataset(
            fold='training',
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
        )

        self.val_dataset = NMPDataset(
            fold='validation',
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            batch_size=batch_size,
        )

        self.test_dataset = NMPDataset(
            fold='test',
            audio_dir=audio_dir,
            meta_path=meta_path,
            sample_rate=sample_rate,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            nn_context_dim=nn_context_dim,
            ar_context_dim=ar_context_dim,
            ar_order=ar_order,
            ar_fade_dim=ar_fade_dim,
            diagonal_load=diagonal_load,
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, **self.kwargs)
