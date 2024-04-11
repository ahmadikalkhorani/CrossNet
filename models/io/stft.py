import torch
from torch import nn

from torch import Tensor
from typing import *

paras_16k = {
    'n_fft': 512,
    'n_hop': 256,
    'win_len': 512,
}

paras_8k = {
    'n_fft': 256,
    'n_hop': 128,
    'win_len': 256,
}


class STFT(nn.Module):

    def __init__(self, n_fft: int, n_hop: int, win_len: Optional[int] = None, win: str = 'hann_window') -> None:
        super().__init__()
        win_len = win_len if win_len is not None else n_fft
        self.n_fft, self.n_hop, self.win_len = n_fft, n_hop, win_len
        self.repr = str((n_fft, n_hop, win, win_len))

        assert win == 'hann_window', win
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

    def forward(self, X: Tensor, original_len: int = None, inverse=False) -> Any:
        """istft
        Args:
            X: complex [..., F, T]
            original_len: original length
            inverse: stft or istft
        """
        if not inverse:
            return self.stft(X)
        else:
            return self.istft(X, original_len=original_len)

    def stft(self, x: Tensor) -> Tuple[Tensor, int]:
        """stft
        Args:
            x: [..., time]

        Returns:
            the complex STFT domain representation of shape [..., freq, time] and the original length of the time domain waveform
        """
        shape = list(x.shape)
        x = x.reshape(-1, shape[-1])
        if x.is_cuda:
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):  # use float32 for stft & istft
                X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, return_complex=True)
        else:
            X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, return_complex=True)
        F, T = X.shape[-2:]
        X = X.reshape(shape=shape[:-1] + [F, T])
        return X, shape[-1]

    def istft(self, X: Tensor, original_len: int = None) -> Tensor:
        """istft
        Args:
            X: complex [..., F, T]
            original_len: returned by stft

        Returns:
            the complex STFT domain representation of shape [..., freq, time] and the original length of the time domain waveform
        """
        shape = list(X.shape)
        X = X.reshape(-1, *shape[-2:])
        if X.is_cuda:
            with torch.autocast(device_type=X.device.type, dtype=torch.float32):  # use float32 for stft & istft
                x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, length=original_len)
        else:
            x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, length=original_len)
        x = x.reshape(shape=shape[:-2] + [original_len])
        return x

    def extra_repr(self) -> str:
        return self.repr


if __name__ == '__main__':
    x = torch.randn((1, 1, 8000 * 4))
    stft = STFT(**paras_8k)
    X, ol = stft.stft(x)
    x_p = stft.istft(X, ol)
    print(x.shape, x_p.shape, X.shape)
    print(torch.allclose(x, x_p, rtol=1e-1))
    print()