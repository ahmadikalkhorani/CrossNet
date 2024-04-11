
from typing import *
import torch 
import torchaudio 
import matplotlib.pyplot as plt

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
        noisy = torchaudio.functional.add_noise(waveform = audio, noise = noise, snr = torch.tensor([snr]))
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
        speed_factor = randfloat(rng, *self.spd_factors)
        audio = torchaudio.functional.speed(audio, orig_freq=self.fs, factor=speed_factor)[0]
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

    def transform(self, x, rng):
        for t in self.transforms:
            x = t(x, rng)
        return x

if __name__ == "__main__":
    transorms = Compose(
        [
            RandomApply(Speed(fs = 8000, speed_factors=[0.95, 1.05]), p = 1.0),
            RandomApply(WhiteNoise(min_snr=10, max_snr=50), p = 0.3),
            RandomApply(PolarityInversion(), p = 0.5),
        ]
    )
    s, sr = torch.sin(3*2*3.1415*torch.linspace(0, 1, 8000)).reshape(1, -1), 8000
    rng = torch.Generator()
    noise = torch.empty(s.shape).normal_(mean=0.0,std=1, generator = rng)
    plt.plot(s.reshape(-1))
    # plt.plot(noise.reshape(-1))
    noisy = transorms(s, rng)
    plt.plot(s.reshape(-1))
    plt.plot(noisy.reshape(-1), label = "transorm")
    plt.legend()