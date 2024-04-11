
from typing import *
import torch 
import torchaudio 
import matplotlib.pyplot as plt
import os
from torch import Tensor
from torch.utils.data import Dataset
import torch
import numpy as np
import soundfile as sf
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.signal import convolve, resample
import json
from typing import Callable, Dict, List, Optional, Tuple, Union
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
import random 
import pandas as pd
from torch.utils.data.distributed import DistributedSampler, T_co
from typing import TypeVar, Optional, Iterator
import math 
import glob
from pathlib import Path 

FS_ORIG = 16000 



def randint(g: torch.Generator, low: int, high: int) -> int:
    """return [low, high)
    """
    r = torch.randint(low=low, high=high, size=(1,), generator=g, device='cpu')
    return r[0].item()  # type:ignore

def randfloat(g: torch.Generator, low: float, high: float) -> float:
    """return [low, high)
    """
    r = torch.rand(size=(1,), generator=g, device='cpu')[0].item()
    return float(low + r * (high - low))

def randchoice(g: torch.Generator, seq: List[Any]) -> Any:
    return seq[ randint(g, 0, len(seq)) ]

class RandomApply(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, x, rng):
        if self.p < randfloat(rng, 0.0, 1.0):
            return x
        x = self.transform(x, rng)
        return x


class WhiteNoise(torch.nn.Module):
    def __init__(self, min_snr=10, max_snr=50):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio, rng):
        snr = randfloat(rng, self.min_snr, self.max_snr)
        noise = torch.empty(audio.shape, dtype = audio.dtype).normal_(mean=0.0,std=1.0, generator = rng)
        noisy = torchaudio.functional.add_noise(waveform = audio, noise = noise.reshape(*audio.shape), snr = torch.tensor([snr]))
        return noisy

class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio, rng):
        audio = torch.neg(audio)
        return audio

class Speed(torch.nn.Module):
    def __init__(self, fs: int, speed_factors: List[float] = [0.95, 1.05], ):
        super().__init__()
        self.fs = fs
        self.spd_factors = speed_factors
        
    def forward(self, audio, rng): 
        T = audio.shape[-1]
        speed_factor = randfloat(rng, *self.spd_factors)
        audio = torchaudio.functional.speed(audio, orig_freq=self.fs, factor=speed_factor)[0]
        
        if audio.shape[-1] > T:
            audio = audio[..., :T]
        elif audio.shape[-1] < T:
            audio = torch.cat([audio, torch.zeros(*audio.shape[:-1], T-audio.shape[-1])], dim = -1)
        return audio
    

class PitchShift(torch.nn.Module):
    def __init__(self, fs: int, n_steps: List[int] = [4,10], ):
        super().__init__()
        self.fs = fs
        self.n_steps = n_steps
        
    def forward(self, audio, rng):
        pitch_factor = randint(rng, *self.n_steps)
        audio = torchaudio.functional.pitch_shift(audio, self.fs, pitch_factor)
        return audio

class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, rng):
        x = self.transform(x, rng)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x, rng = None):
        if rng is None:
            rng = torch.Generator() 
            rng.manual_seed(np.random.randint(0, 99999999999))
        for t in self.transforms:
            x = t(x, rng)
        return x






def chunk_audio(x, g, ratio = [0.7, 1.0]):
    # makes anything outside a random chunk zero
    r = randfloat(g, ratio[0], ratio[1])
    start = randint(g=g, low = 0, high = int((1-r)*x.shape[-1]))
    end = start + int(r * x.shape[-1])
    
    x[..., :start] = 0.0
    x[..., end:] = 0.0
    
    return x







class SS_SemiOnlineDataset(Dataset):
    """A semi-online dataset for speech separation: dynamicly convolve RIRs and speech pairs
    """

    @staticmethod
    def collate_fn(batches):
        mini_batch = []
        for x in zip(*batches):
            if isinstance(x[0], np.ndarray):
                x = [torch.tensor(x[i]) for i in range(len(x))]
            if isinstance(x[0], Tensor):
                x = torch.stack(x)
            mini_batch.append(x)
        return mini_batch

    def __init__(self,
                 speeches: List[List[Dict[str, str]]],
                 sample_rate: int,
                 audio_time_len: Optional[float] = None,
                 dynamic_mix = False,
                 raw_wsj_cfg: dict = None,
                 raw_wsj0_path: str = None,
                 ordering: str = None
                 ) -> None:
        """initialze 

        Args:
            speeches: paths of single channel clean speech pairs
            audio_time_len: audio signal length (in seconds). Shorter signals will be appended zeros, longer signals will be cut to the length
            sample_rate: sample rate. if specified, signals will be downsampled or upsampled.
        """
        
        self.speeches = speeches
        self.sample_rate = sample_rate
        self.audio_time_len = float(audio_time_len) if audio_time_len is not None else None
        self.dm = dynamic_mix
        self.raw_wsj_cfg = raw_wsj_cfg 
        self.raw_wsj0_path = raw_wsj0_path
        self.order = ordering
        
        assert ordering in [None, "pitch"], "ordering should be None or pitch: "+ str(ordering)
        
        print(f"raw_wsj0_path: {self.raw_wsj0_path}")
        
        self._seed = 0
        
        self.g = torch.Generator()
        
        self.total_speech = speeches[0] + speeches[1]
        
        if self.dm:
            self.transforms = Compose(
                [
                    RandomApply(Speed(fs = self.sample_rate, speed_factors=[0.95,1.05]), p = 1.0),
                    RandomApply(WhiteNoise(min_snr=10, max_snr=50), p = 0.3),
                    RandomApply(PolarityInversion(), p = 0.5),
                ]
            )
    

            
         
          
    def seed(self):
        self._seed += 1
        return self._seed 
    
    
    
    def pad(self, x: torch.Tensor, N: int): 
        # x: [channel, time]
        if x.shape[-1] < N:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], N - x.shape[-1])], dim = -1)
        elif x.shape[-1] > N:
            start = np.random.randint(low=0, high=x.shape[-1] - N)
            x = x[..., start:start + N]
        

        return x
    
    def read_chunk(self, wav_path, N, g: torch.Generator):
        meta_data = torchaudio.info(wav_path)
        assert meta_data.sample_rate == self.sample_rate 
        
        num_frames = meta_data.num_frames
        start = 0
        if num_frames > N:
            start = randint(g, 0, num_frames - N)
        
        audio, sr = torchaudio.load(wav_path, num_frames = N, frame_offset = start)
        if audio.shape[-1] < N:
            audio = torch.nn.functional.pad(audio, (0, N - audio.shape[-1]))
    
        return audio
            
            

    def dynamic_mix(self, index):
        g = self.g
        g.manual_seed(index['dynamic_seed'])
        
        samplerate = self.sample_rate
        wsj0_path = self.raw_wsj0_path
        
        cfgs = [randchoice(g, self.raw_wsj_cfg) for _ in range(2)]
        # cfgs = np.random.choice(self.raw_wsj_cfg, 2)
        # avoid same speaker mixtures
        while(cfgs[0]["speaker"] == cfgs[1]["speaker"]):
            cfgs = [randchoice(g, self.raw_wsj_cfg) for _ in range(2)]
            
        N = int(self.audio_time_len*self.sample_rate)

        sources = [self.read_chunk(str(Path(wsj0_path) / cfg["file"]), N = N, g = g, ) for cfg in cfgs]
                
        snr = randfloat(g, 0, 2.5)
        snrs = [snr, -snr]


        activelev_scales = [ cfg["active_level"] for cfg in cfgs]        
        scaled_sources = [s / np.sqrt(scale) for s, scale in zip(sources, activelev_scales)]
        
        sources= [ self.transforms(s, rng = g, ) for s in scaled_sources] 
        
        scaled_sources = [s  * 10 ** (snr/20) for s, snr in zip(sources, snrs)]
                                
        sources = torch.stack(sources)
        
        mix = sources.sum(dim = 0)
        
        gain = np.max([1., torch.max(torch.abs(mix)), torch.max(torch.abs(sources))]) / 0.9
        x = mix / gain
        y = sources / gain
        
        
  
        
        p = {
            **index,
            **cfgs[0],
            **cfgs[1],
            "sample_rate": self.sample_rate,
            "s1": cfgs[0]["file"],
            "s2": cfgs[1]["file"],
            "snrs": snrs, 
            }
        
        if "01gc020m.wav" in cfgs[0]["file"]:
            print(p)
        
        
        return x, y, p 
    
    def __getitem__(self, index: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:  # type: ignore
        """returns the indexed item

        Args:
            index: index

        Returns:
            Tensor: xm of shape [channel, time]
            Tensor: ys of shape [spk, channel, time]
            dict: paras used
        """
        try:
            if self.dm:
                x, y, p = self.dynamic_mix(index)
            else:
                sidx = index['speech_index']
                
                
                s1, sr1 = self.read(self.speeches[0][sidx])
                s2, sr2 = self.read(self.speeches[1][sidx])
                mix, srmix = self.read(self.speeches[2][sidx])
                
                s1 = s1.reshape(1, -1)
                s2 = s2.reshape(1, -1)
                mix = mix.reshape(1, -1)
                
                start, end = 0, None
                if self.audio_time_len is not None:
                    # g = torch.Generator()
                    # g.manual_seed(index['seed'] + self.seed() if "cv" not in self.speeches[0][sidx] else index['seed'])

                    N = int(self.audio_time_len*self.sample_rate)
                    L = s1.shape[-1]
                    
                    if L < N:
                        s1 = torch.nn.functional.pad(s1,   (0, N - L))
                        s2 = torch.nn.functional.pad(s2,   (0, N - L))
                        mix = torch.nn.functional.pad(mix, (0, N - L))
                    elif L > N:
                        start = np.random.randint(low=0, high=L - N)
                        end = start + N
                
                
                s1 = s1[:, start:end]
                s2 = s2[:, start:end]
                mix = mix[:, start:end]
                
                x = mix
                y = torch.stack([s1, s2])
                
                p = {
                        "index": sidx,
                        "sample_rate": self.sample_rate,
                        "s1": self.speeches[0][sidx],
                        "s2": self.speeches[1][sidx],
                        "mix": self.speeches[2][sidx],
                        "num_samples": x.shape[-1], 
                    }
            
            
            
            return x, y, p
        except Exception as e:
            print(e)
            return self.__getitem__({
                "speech_index": np.random.randint(0, len(self.speeches[0])), 
                "seed": np.random.randint(0, 999999999),
                "dynamic_seed": np.random.randint(0, 999999999),
                })      

    def __len__(self):
        return self.speech_num()

    def speech_num(self):
        return len(self.speeches[0])

    def rir_num(self):
        return len(self.rirs)

    def read(self, wav_path, N = None, g = None):
        
        clean, samplerate = sf.read(wav_path, dtype='float32')
        assert len(clean.shape) == 1, "clean speech should be single channel"
        # resample if necessary
        if self.sample_rate != None and samplerate != self.sample_rate:
            re_len = int(clean.shape[0] * self.sample_rate / samplerate)
            clean = resample(clean, re_len)
        
        clean = torch.from_numpy(clean)
        
        if self.dm: # data augmentation
            clean = self.transforms(clean.reshape(1, -1), rng = g).reshape(-1)
            
        if self.audio_time_len is not None:
            # pad if small
            if len(clean) < (self.audio_time_len*self.sample_rate):
                clean = torch.nn.functional.pad(clean, (0, int(self.audio_time_len * self.sample_rate) -len(clean)))
        
        
        if N is not None:
            start = randint(g, low=0, high=len(clean) - N)
            end = start + N 
            clean = clean[..., start:end]
            
        return clean, samplerate



class SS_SemiOnlineSampler(DistributedSampler[T_co]):
    r"""Sampler for SS_SemiOnlineDataset for single GPU and multi GPU (or Distributed) cases.
    If shuffle == True, the speech pair sequence and seed changes along epochs, else the speech pair and seed won't change along epochs
    If shuffle_rir == True, the rir will be shuffled, otherwise not

    No matter what is ``shuffle`` or ``shuffle_rir``, the speech sequence, rir sequence, seed generated for dataset are all deterministic.
    They all determined by the parameter ``seed`` and ``epoch`` (for shuffle == True)
    """

    def __init__(self,
                 dataset: SS_SemiOnlineDataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        try:
            super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        except:
            # if error raises, it is running on single GPU
            # thus, set num_replicas=1, rank=0
            super().__init__(dataset, 1, 0, shuffle, seed, drop_last)
        self.last_epoch = -1

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            speech_indices = torch.randperm(self.dataset.speech_num(), generator=g).tolist()  # type: ignore
          

            if self.last_epoch >= self.epoch:
                import warnings
                if self.epoch != 0:
                    warnings.warn('the epoch value doesn\'t update when shuffle is true, the training data and sequence won\'t change along with different epochs')
            else:
                self.last_epoch = self.epoch
        else:
            g = torch.Generator()
            g.manual_seed(self.seed)
            speech_indices = list(range(len(self.dataset)))  # type: ignore
        
        # make rir_indices and speech_indices have the same length as the dataset
        if len(speech_indices) > len(self.dataset):  # type: ignore
            speech_indices = speech_indices[:len(self.dataset)]  # type: ignore

        # construct indices
        indices = []
        for sidx in speech_indices:
            seed = torch.randint(high=9999999999, size=(1,), generator=g)[0].item()
            indices.append({'speech_index': sidx, 'seed': seed, "dynamic_seed": seed + self.epoch})

        # drop last
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)  # type: ignore

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class SS_SemiOnlineDataModule(LightningDataModule):
    """A semi-online DataModule about how to prepare data, construct dataset, and dataloader
    """
    rirs: Dict[str, List[str]]
    spk1_cfgs: Dict[str, List[Dict[str, str]]]
    spk2_cfgs: Dict[str, List[Dict[str, str]]]

    def __init__(
        self,
        clean_speech_dir: str,
        sample_rate: int = 8000,
        batch_size: List[int] = [5, 5],
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],
        num_workers: int = 15,
        collate_func_train: Callable = SS_SemiOnlineDataset.collate_fn,
        collate_func_val: Callable = SS_SemiOnlineDataset.collate_fn,
        collate_func_test: Callable = SS_SemiOnlineDataset.collate_fn,
        test_set: str = 'test',
        seeds: Dict[str, Optional[int]] = {
            'train': None,
            'val': 2,  # fix val and test seeds to make sure they won't change in any time
            'test': 3
        },
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor=5,
        audio_only: bool = True,
        dynamic_mix = False,
        config_dir = "configs", 
        ordering = None,
        
    ):

        super().__init__()

        self.test_set = test_set
        
        self.sample_rate = sample_rate

        self.audio_only = audio_only
        
        self.config_dir = config_dir
        
        self.ordering = ordering

        self.seeds: Dict[str, int] = dict()
        for k, v in seeds.items():
            if v is None:
                v = random.randint(0, 1000000)
            self.seeds[k] = v
        print('seeds for datasets:', self.seeds)
        # generate seeds for train, validation and test
        # self.seed = seed
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)
        # self.seeds = {'train': randint(self.g, 0, 100000), 'val': randint(self.g, 0, 100000), 'test': randint(self.g, 0, 100000)}


        self.clean_speech_dir = os.path.expanduser(clean_speech_dir)


        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]
        rank_zero_info(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test={self.batch_size_test}')

        self.num_workers = num_workers

        self.audio_time_len = audio_time_len[0]
        self.audio_time_len_for_val = None if len(audio_time_len) < 2 else audio_time_len[1]
        self.audio_time_len_for_test = None if len(audio_time_len) < 3 else audio_time_len[2]
        rank_zero_info(f'audio_time_len: train={self.audio_time_len}; val={self.audio_time_len_for_val}; test={self.audio_time_len_for_test}')

        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        
        self.speech_dir_map = {"train": "tr", "validation": "cv", "test": "tt" }
        
        self.dynamic_mix = dynamic_mix


            
        self.prepare_data()
            

    def prepare_data(self):
        
        with open(f"{self.config_dir}/datasets/wsj_cfg.json", "r") as f:
            raw_wsj_cfg = json.load(f)
            
        for ds in raw_wsj_cfg.keys():
            for i in range(len(raw_wsj_cfg[ds])):
                raw_wsj_cfg[ds][i]["file"] = os.path.join(self.clean_speech_dir, "wsj0_8k", raw_wsj_cfg[ds][i]["file"]) 
        self.raw_wsj_cfg = raw_wsj_cfg

        # self.rirs: Dict[str, List[str]]
        self.speeches = {
            "train": [
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tr/s1/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tr/s2/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tr/mix/*.wav", recursive=True),
            ],
            "validation": [
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/cv/s1/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/cv/s2/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/cv/mix/*.wav", recursive=True),
            ],
            "test": [
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tt/s1/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tt/s2/*.wav", recursive=True),
                glob.glob(f"{self.clean_speech_dir}/wsj0-mix/2speakers/wav8k/min/tt/mix/*.wav", recursive=True),
            ],
        }


    def setup(self, stage=None):
    
        self.train = SS_SemiOnlineDataset(
            speeches= self.speeches["train"],
            audio_time_len=self.audio_time_len,
            sample_rate = self.sample_rate,
            dynamic_mix = self.dynamic_mix,
            raw_wsj_cfg = self.raw_wsj_cfg["train"],
            raw_wsj0_path=os.path.join(self.clean_speech_dir, "wsj0"),
            ordering=self.ordering, 
        )
        self.val = SS_SemiOnlineDataset(
            speeches= self.speeches["validation"],
            audio_time_len=self.audio_time_len_for_val,
            sample_rate = self.sample_rate,
            dynamic_mix = False,
            ordering=None, 
        )
        self.test = SS_SemiOnlineDataset(
            speeches= self.speeches["test"],
            audio_time_len=None,
            sample_rate = self.sample_rate,
            dynamic_mix = False,
            ordering=None, 
        )

    def train_dataloader(self) -> DataLoader:
        persistent_workers = False
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_func_train,
                          sampler=SS_SemiOnlineSampler(self.train, seed=self.seeds['train'], shuffle=True, ),
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          pin_memory=self.pin_memory,
                          persistent_workers=persistent_workers, 
                          worker_init_fn = lambda x: np.random.seed(self.seeds["train"] + x),
                          )

    def val_dataloader(self) -> DataLoader:
        persistent_workers = False
        return DataLoader(self.val,
                          batch_size=self.batch_size_val,
                          collate_fn=self.collate_func_val,
                          sampler=SS_SemiOnlineSampler(self.val, seed=self.seeds['val'], shuffle=False, ),
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          pin_memory=self.pin_memory,
                          persistent_workers=persistent_workers, 
                          worker_init_fn = lambda x: np.random.seed(self.seeds["val"] + x), 
                          )

    def test_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            collate_fn=self.collate_func_test,
            sampler=SS_SemiOnlineSampler(dataset, seed=self.seeds['test'], shuffle=False, ),
            num_workers=3,
            prefetch_factor=prefetch_factor,
            worker_init_fn = lambda x: np.random.seed(self.seeds["test"] + x),
        )

    def predict_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=SS_SemiOnlineDataset.collate_fn,
            sampler=SS_SemiOnlineSampler(dataset, seed=self.seeds['test'], shuffle=False, shuffle_rir=True),
            num_workers=3,
            prefetch_factor=prefetch_factor,
        )







