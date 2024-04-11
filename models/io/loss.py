from torch import Tensor
import torch
from torch import nn
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr
from models.io.pit import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate as permutate
from models.io.stft import STFT
from typing import *
import torch.nn.functional as F
import torchaudio

def neg_sa_sdr(preds: Tensor, target: Tensor, scale_invariant: bool = False, **kwargs) -> Tensor:
    batch_size = target.shape[0]
    sa_sdr_val = sa_sdr(preds=preds, target=target, scale_invariant=scale_invariant)
    return -torch.mean(sa_sdr_val.view(batch_size, -1), dim=1)


def neg_si_sdr(preds: Tensor, target: Tensor, **kwargs) -> Tensor:
    """calculate neg_si_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)


def neg_snr(preds: Tensor, target: Tensor, **kwargs) -> Tensor:
    """calculate neg_snr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    snr_val = snr(preds=preds, target=target)
    return -torch.mean(snr_val.view(batch_size, -1), dim=1)


def scale_invariant_signal_distortion_ratio(preds: torch.Tensor, target: torch.Tensor, scale_target = False, multiplier: float = -1.0, **kwargs) -> torch.Tensor:
    """
    TFGridNet eq 9
    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not
   
    """
    # with torch.autocast(device_type = "cuda", enabled = False):
            
    #     preds = preds.to(torch.float32)
    #     target = target.to(torch.float32)
        
    eps = torch.finfo(preds.dtype).eps

    if scale_target:
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / ( torch.sum(target**2, dim=-1, keepdim=True) + eps )
        target_scaled = alpha * target
        preds_scaled = preds
    else:
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / ( torch.sum(preds**2, dim=-1, keepdim=True) + eps )
        target_scaled = target
        preds_scaled = alpha * preds

    noise = target_scaled - preds_scaled

    val = torch.sum(target_scaled**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
    val = 10 * torch.log10(val)
    val = multiplier * val

    return val.mean(-1) # return [B]

def wav_mag(preds, target, stft_func, scale_invariant = True, **kwargs):
    ''' 
    preds: [B, spk, time]
    target: [B, spk, time]
    '''
    
    assert preds.shape == target.shape, "preds and target should have same shape"
    assert preds.ndim == 3, "preds and target should be 3D tensor"
    

    preds_stft  = torch.abs(stft_func(preds)[0]).flatten(start_dim=2) # [B, spk, F, T]
    target_stft  = torch.abs(stft_func(target)[0]).flatten(start_dim=2) # [B, spk, F, T]

    wav_loss = torch.linalg.norm( preds - target, dim = -1, ord = 1 )
    stft_loss = torch.linalg.norm( preds_stft - target_stft, dim = -1, ord = 1 )
    
    if scale_invariant:
        stft_loss = stft_loss / torch.clamp(torch.linalg.norm( target_stft, dim = -1, ord = 1 ), min=1e-5)
        wav_loss = wav_loss / torch.clamp(torch.linalg.norm( target, dim = -1, ord = 1 ), min=1e-5) 
    else:
        stft_loss = stft_loss / preds_stft.shape[-1]
        wav_loss  = wav_loss  / preds.shape[-1] 
    
    loss = stft_loss + wav_loss
    
    if loss.ndim == 2:
        loss = loss.mean(dim = -1)

    return loss


def RIMag(preds, target, stft_func,  **kwargs):
    ''' 
    preds: [B, spk, time]
    target: [B, spk, time]
    '''
    
    assert preds.shape == target.shape, "preds and target should have same shape"
    assert preds.ndim == 3, "preds and target should be 3D tensor"
    
    preds_stft  = stft_func(preds)[0]   # [B, spk, F, T]
    target_stft  = stft_func(target)[0] # [B, spk, F, T]
    
    R = torch.abs(preds_stft.real-target_stft.real) 
    I = torch.abs(preds_stft.imag-target_stft.imag)
    Mag = torch.abs(torch.abs(preds_stft)-torch.abs(target_stft))
 
    loss = torch.mean( R + I, dim=[-2,-1])
    loss += torch.mean(Mag, dim = [-2, -1])
    
    if loss.ndim == 2:
        loss = loss.mean(dim = -1) 
    
    return loss





def mixture_constraint(preds, target, loss_func, **kwargs):
    ''' 
    preds: [B, spk, time]
    target: [B, spk, time]
    '''
    
    assert preds.shape == target.shape, "preds and target should have same shape"
    assert preds.ndim == 3, "preds and target should be 3D tensor"


    preds_mix = torch.sum(preds, dim = 1, keepdim=True) # [B, 1, time]
    target_mix = torch.sum(target, dim = 1, keepdim=True) # [B, 1, time]
    
    return loss_func(preds = preds_mix, target = target_mix, **kwargs)

    # time domain loss
    










class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        delta_win_length=5

        loss_1 = torch.norm(y_mag - x_mag, p="fro", dim = (-2, -1), ) / torch.clamp(torch.norm(y_mag, p="fro", dim = (-2, -1), ), min=1e-5)

        x_mag=x_mag.transpose(-1,-2)
        y_mag=y_mag.transpose(-1,-2)

        x_del = torchaudio.functional.compute_deltas(x_mag,win_length=delta_win_length)
        y_del = torchaudio.functional.compute_deltas(y_mag,win_length=delta_win_length)
        
        loss_2 = torch.norm(y_del - x_del, p="fro", dim = (-2, -1), ) / torch.clamp(torch.norm(y_del, p="fro", dim = (-2, -1), ), min=1e-5)
        
        x_acc = torchaudio.functional.compute_deltas(x_del,win_length=delta_win_length)
        y_acc = torchaudio.functional.compute_deltas(y_del,win_length=delta_win_length)
        loss_3 = torch.norm(y_acc - x_acc, p="fro", dim = (-2, -1), ) / torch.clamp(torch.norm(y_acc, p="fro", dim = (-2, -1), ), min=1e-5)
        

        return loss_1 + loss_2 + loss_3
    

class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        delta_win_length=5
        
        loss_1 = F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduce=False)
        loss_1 = torch.mean(loss_1, dim = (-2, -1))

        x_mag=torch.log(x_mag).transpose(-1,-2)
        y_mag=torch.log(y_mag).transpose(-1,-2)
        
        x_del = torchaudio.functional.compute_deltas(x_mag ,win_length=delta_win_length)
        y_del = torchaudio.functional.compute_deltas(y_mag ,win_length=delta_win_length)
        loss_2 = F.l1_loss(y_del, x_del, reduce=False)
        loss_2 = torch.mean(loss_2, dim = (-2, -1))
        
        
        x_acc = torchaudio.functional.compute_deltas(x_del ,win_length=delta_win_length)
        y_acc = torchaudio.functional.compute_deltas(y_del ,win_length=delta_win_length)
        loss_3 = F.l1_loss(y_acc, x_acc, reduce=False)
        loss_3 = torch.mean(loss_3, dim = (-2, -1))


        return loss_1 + loss_2 + loss_3
    
class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))
    
    def stft(self, x, fft_size, hop_size, win_length, window):
        """Perform STFT and convert to magnitude spectrogram.

        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window lengtorch.
            window (str): Window function type.

        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

        """
        # window = window.to(x.device)
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)

        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-5)).transpose(2, 1)

    def forward(self, x, y, **kwargs):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = self.stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = self.stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss 
    
class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes= [120, 240, 50],
        win_lengths= [600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        
        self.__name__ = "MultiResolutionSTFTLoss"
        

    def forward(self, preds, target, **kwargs):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        
        x = preds 
        y = target
        
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T) 
        
        negsidr = neg_si_sdr(x, y) 
        # negsidr = 0.0
        
        
        
        sc_loss = 0.0
        mag_loss = 0.0
        for i, f in enumerate(self.stft_losses):
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)


        return sc_loss + mag_loss + negsidr




class Loss(nn.Module):
    is_scale_invariant_loss: bool
    name: str

    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict(), ):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
            neg_si_sdr: True,
            neg_snr: False,
            wav_mag: False,
        }.get(loss_func, "False")
        
        self.name = loss_func.__name__

        
        self.mixture_constraint_loss = False
        if "mixture_constraint" in loss_func_kwargs.keys():
            if loss_func_kwargs["mixture_constraint"]:
                self.mixture_constraint_loss = True
                print("Adding mixture constraint loss")


    def forward(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        t_min = min(preds.shape[-1], target.shape[-1])
        preds, target = preds[..., :t_min], target[..., :t_min]
        
        perms = None
        if self.pit:
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode="permutation-wise", **self.loss_func_kwargs, **kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs, **kwargs)
            
        if self.mixture_constraint_loss:
            preds_mix = torch.sum(preds, dim = 1, keepdim=True) # [B, 1, time]
            target_mix = torch.sum(target, dim = 1, keepdim=True) # [B, 1, time]
    
            losses = losses + self.loss_func(preds=preds_mix, target=target_mix, **self.loss_func_kwargs, **kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds
    
    def test(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        t_min = min(preds.shape[-1], target.shape[-1])
        preds, target = preds[..., :t_min], target[..., :t_min]
        
        perms = None
        if True: # always use pit in test
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode="permutation-wise", **self.loss_func_kwargs, **kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
     
            
        if self.mixture_constraint_loss:
            losses = losses + mixture_constraint(preds = preds, target = target, loss_func = self.loss_func, **self.loss_func_kwargs, **kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}"
