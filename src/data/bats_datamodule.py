import torch
import torchaudio
import polars as pl
import os
from typing import Dict, Tuple, Optional, Any, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = [wav.shape[-1] for wav in waveforms]

    prepared_waveforms = [wav.squeeze() for wav in waveforms]
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(prepared_waveforms, batch_first=True)
    return padded_waveforms, torch.tensor(labels), torch.tensor(lengths)


class BatsDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        sample_rate: int = 16000,
        desired_length: int = 2,
        n_fft: int = 510,
        hop_length: int = 125,
        n_mels: int = 64,
        use_mel: bool = False,
        *args,
        **kwargs,
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.len_threshold = desired_length * sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_mel = use_mel

        self.transform = self._create_transform()

    def _create_transform(self):
        transforms = []
        if self.use_mel:
            transforms.append(
                torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            )
        else:
            transforms.append(torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length))
        transforms.append(torchaudio.transforms.AmplitudeToDB())
        transforms.append(torchaudio.transforms.TimeMasking(time_mask_param=80))
        transforms.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=80))
        return torch.nn.Sequential(*transforms)

    def low_pass(self, waveform, cutoff=100000):
        """Keep frequencies below cutoff"""
        return torchaudio.functional.lowpass_biquad(
            waveform, 
            self.sample_rate, 
            cutoff_freq=cutoff
        )

    def high_pass(self, waveform, cutoff=100):
        """Remove frequencies below cutoff"""
        return torchaudio.functional.highpass_biquad(
            waveform,
            self.sample_rate,
            cutoff_freq=cutoff
        )

    def band_pass(self, waveform, low_cut=20000, high_cut=100000):
        """Isolate frequency band"""
        waveform = self.high_pass(waveform, low_cut)
        return self.low_pass(waveform, high_cut)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        waveform, sr = torchaudio.load(self.filepaths[idx])

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[-1] < self.len_threshold:
            pad = (0, self.len_threshold - waveform.shape[-1])
            waveform = torch.nn.functional.pad(waveform, pad)
        else:
            waveform = waveform[..., : self.len_threshold]

        #waveform = self.band_pass(waveform)
        spectrogram = self.transform(waveform)

        # Normalize spectrogram
        #spectrogram = self._normalize(spectrogram)

        # Add channel dimension (if needed) and flatten
        # spectrogram = spectrogram.unsqueeze(0)  # [1, n_mels, time]
        spectrogram = spectrogram.repeat(3, 1, 1)
        return spectrogram, self.labels[idx]


class BatsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        annotations_fname: str,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        sample_rate: int = 250000,
        pin_memory: bool = False,
        n_fft: int = 510,
        hop_length: int = 125,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # self.transforms = transforms.Compose(

        # )

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10  # simplified dataset contains that much classes.

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass  # for now we assume the data is downloaded (and prepared).

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        annotations = pl.read_csv(os.path.join(self.hparams.data_dir, self.hparams.annotations_fname))
        emmiters = (
            annotations.select(
                [
                    pl.col("Emitter").rank("dense") - 1,
                ]
            )
            .to_torch()
            .ravel()
        )
        filepaths = [
            os.path.join(self.hparams.data_dir, folder_name, file_name)
            for folder_name, file_name in zip(annotations["File folder"], annotations["File name"])
        ]
        dataset = BatsDataset(
            filepaths=filepaths,
            labels=emmiters,
            **self.hparams,
        )

        lengths = [int(len(dataset) * ratio) for ratio in self.hparams.train_val_test_split[:-1]]
        lengths.append(len(dataset) - sum(lengths))
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(245),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=collate_fn,
        )

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = BatsDataModule()
