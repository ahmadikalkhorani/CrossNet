from typing import Dict, List, Optional, Tuple, Union
import warnings
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.audio import *
from torchmetrics.functional.audio import *
from torch import Tensor
import torch
import pesq as pesq_backend
import numpy as np
from typing import *
import warnings
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.audio import *
from torchmetrics.functional.audio import *
from torch import Tensor
import torch
import pesq as pesq_backend
import numpy as np
from typing import *
import traceback
from models.utils.asr_metric import Whisper
from pypesq import pesq as pypesq

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI']

# import speechmetrics 

class MOS(torch.nn.Module):
    def __init__(self, fs = 8000, ): 
        super(MOS, self).__init__()
        self.sm = speechmetrics.load("mosnet", window = 10)
        self.fs = fs
    
    def calculate(self, x, fs):
        return torch.from_numpy(self.sm(x, rate = fs)["mosnet"])
    
    def forward(self, preds, fs):
        # preds: spk x time 
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        
        preds = preds.detach().cpu().numpy()
        
        assert preds.ndim == 2, "preds should be 2d array - spk x time"

        return torch.cat([self.calculate(preds[i], fs = fs) for i in range(preds.shape[0])])


# speechmetrics_mos = MOS()

def construct_audio_MetricCollection(
    metrics: List[str],
    prefix: str = '',
    postfix: str = '',
    fs: int = None,
) -> MetricCollection:
    md: Dict[str, Metric] = {}
    for m in metrics:
        mname = prefix + m.lower() + postfix
        if m.upper() == 'SDR':
            md[mname] = SignalDistortionRatio()
        elif m.upper() == 'SI_SDR':
            md[mname] = ScaleInvariantSignalDistortionRatio()
        elif m.upper() == 'SI_SNR':
            md[mname] = ScaleInvariantSignalNoiseRatio()
        elif m.upper() == 'SNR':
            md[mname] = SignalNoiseRatio()
        elif m.upper() == 'NB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='nb').cpu()
        elif m.upper() == 'WB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='wb').cpu()
        elif m.upper() == 'STOI':
            md[mname] = ShortTimeObjectiveIntelligibility(fs).cpu()
        else:
            raise ValueError('Unkown audio metric ' + m)

    return MetricCollection(md)


def cal_metrics(
    preds: Tensor,
    target: Tensor,
    original: Union[Tensor, Dict[str, Tensor]],
    mc: MetricCollection,
    input_mc: Optional[MetricCollection] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """calculate metrics, input_metrics, imp_metrics

    Args:
        preds: prediction
        target: target
        original: original signal or input_metrics
        mc: MetricCollection 
        input_mc: Input MetricCollection if original signal is given, else None

    Returns:
        metrics, input_metrics, imp_metrics
    """
    metrics = mc(preds, target)
    if isinstance(original, Tensor):
        if input_mc == None:
            raise ValueError('input_mc cannot be None when original signal is given, i.e. original is a Tensor')
        input_metrics = input_mc(original, target)
    else:
        input_metrics = original
    imp_metrics = {}
    for k, v in metrics.items():
        v = v.detach().cpu()
        iv = input_metrics['input_' + k].detach().cpu()
        metrics[k] = v
        input_metrics['input_' + k] = iv
        imp_metrics[k + '_i'] = v - iv

    return metrics, input_metrics, imp_metrics


def get_metric_list_on_device(device: Optional[str]):
    metric_device = {
        None: ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ', 'STOI', 'ESTOI', "WER", "MOS"],
        "cpu": ['NB_PESQ', 'WB_PESQ', 'NB_PESQ_PYPESQ', 'STOI', 'ESTOI'],
        "gpu": ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', "WER", "MOS"],
    }
    return metric_device[device]


def cal_metrics_functional(
        metric_list: List[str],
        preds: Tensor,
        target: Tensor,
        original: Optional[Tensor],
        fs: int,
        device_only: Literal['cpu', 'gpu', None] = None,  # cpu-only: pesq, stoi;
        asr = None, 
        target_transcript: str = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    if device_only is None or device_only == 'cpu':
        preds_cpu = preds.detach().cpu()
        target_cpu = target.detach().cpu()
        original_cpu = original.detach().cpu() if original is not None else None
    else:
        preds_cpu = None
        target_cpu = None
        original_cpu = None

    input_metrics = {}
    metrics = {}
    imp_metrics = {}


    for m in metric_list: 
        try:
            mname = m.lower()
            if m.upper() not in get_metric_list_on_device(device=device_only):
                continue

            if m.upper() == 'SDR':
                ## not use signal_distortion_ratio for it gives NaN sometimes
                metric_func = lambda: signal_distortion_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: signal_distortion_ratio(original, target).detach().cpu()
                # assert preds.dim() == 2 and target.dim() == 2 and original.dim() == 2, '(spk, time)!'
                # metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), preds_cpu.numpy(), False)[0]).mean().detach().cpu()
                # input_metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), original_cpu.numpy(), False)[0]).mean().detach().cpu()
            elif m.upper() == 'SI_SDR':
                metric_func = lambda: scale_invariant_signal_distortion_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: scale_invariant_signal_distortion_ratio(original, target).detach().cpu()
            elif m.upper() == 'MOS':
                metric_func = lambda: speechmetrics_mos(preds, fs = fs)
                input_metric_func = lambda: speechmetrics_mos(original, fs = fs)          
            elif m.upper() == 'SI_SNR':
                metric_func = lambda: scale_invariant_signal_noise_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: scale_invariant_signal_noise_ratio(original, target).detach().cpu()
            elif m.upper() == 'SNR':
                metric_func = lambda: signal_noise_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: signal_noise_ratio(original, target).detach().cpu()
            elif m.upper() == 'WER':
                metric_func = lambda: asr(preds, target_transcript, return_transcript = True)
                input_metric_func = lambda: asr(original, target_transcript, return_transcript = True)
            elif m.upper() == 'NB_PESQ':
                metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'nb', n_processes=0)
                input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'nb', n_processes=0)
            
            elif m.upper() == 'NB_PESQ_PYPESQ':
                metric_func = lambda: torch.Tensor([pypesq(ref = target_cpu.reshape(-1).numpy(), deg = preds_cpu.reshape(-1).numpy(), fs = fs)])
                input_metric_func = lambda: torch.Tensor([pypesq(ref = target_cpu.reshape(-1).numpy(), deg = original_cpu.reshape(-1).numpy(), fs = fs)])
            elif m.upper() == 'WB_PESQ':
                metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'wb', n_processes=0)
                input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'wb', n_processes=0)
            elif m.upper() == 'STOI':
                metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs)
                input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs)
            elif m.upper() == 'ESTOI':
                metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs, extended=True)
                input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs, extended=True)
            else:
                raise ValueError('Unkown audio metric ' + m)

            if m.upper() == 'WB_PESQ' and fs == 8000:
                warnings.warn("There is narrow band (nb) mode only when sampling rate is 8000Hz")
                continue  # Note there is narrow band (nb) mode only when sampling rate is 8000Hz

            if True:
                

                
                if m.upper() == 'WER':
                    transcription, wer = metric_func()
                    m_val = wer.cpu().numpy() 
                    metrics[mname + "_transcription"] = transcription
                else:
                    m_val = metric_func().cpu().numpy()
                
                
                metrics[mname] = np.mean(m_val).item()
                metrics[mname + '_all'] = m_val.tolist()  # _all means not averaged
                
                if original is None:
                    continue
                
                if 'input_' + mname not in input_metrics.keys():
                    if m.upper() == 'WER':
                        # transcription, im_val = input_metric_func() 
                        # im_val = im_val.cpu().numpy()
                        # input_metrics['input_' + mname  + "_transcription"] = transcription
                        im_val = np.array([999])
                    
                    else:
                        im_val = input_metric_func().cpu().numpy() if original is not None else np.nan
                    
                    input_metrics['input_' + mname] = np.mean(im_val).item()
                    input_metrics['input_' + mname + '_all'] = im_val.tolist()

                imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]  # _i means improvement
                imp_metrics[mname + '_all' + '_i'] = (m_val - im_val).tolist()
        except Exception as e: 
            # traceback.print_exception(type(e), e, e.__traceback__)

            warnings.warn(f"Error: {mname} - {e}")
            metrics[mname] = None
            metrics[mname + '_all'] = None
            if 'input_' + mname not in input_metrics.keys():
                input_metrics['input_' + mname] = None
                input_metrics['input_' + mname + '_all'] = None
            imp_metrics[mname + '_i'] = None
            imp_metrics[mname + '_i' + '_all'] = None

    return metrics, input_metrics, imp_metrics


def mypesq(preds: np.ndarray, target: np.ndarray, mode: str, fs: int) -> np.ndarray:
    # 使用ndarray是因为tensor会在linux上会导致一些多进程的错误
    ori_shape = preds.shape
    if type(preds) == Tensor:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    else:
        assert type(preds) == np.ndarray, type(preds)
        assert type(target) == np.ndarray, type(target)

    if preds.ndim == 1:
        pesq_val = pesq_backend.pesq(fs=fs, ref=target, deg=preds, mode=mode)
    else:
        preds = preds.reshape(-1, ori_shape[-1])
        target = target.reshape(-1, ori_shape[-1])
        pesq_val = np.empty(shape=(preds.shape[0]))
        for b in range(preds.shape[0]):
            pesq_val[b] = pesq_backend.pesq(fs=fs, ref=target[b, :], deg=preds[b, :], mode=mode)
        pesq_val = pesq_val.reshape(ori_shape[:-1])
    return pesq_val


def cal_pesq(ys: np.ndarray, ys_hat: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    try:
        if sample_rate == 16000:
            wb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='wb').mean()
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [wb_pesq_val, nb_pesq_val]
        elif sample_rate == 8000:
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [None, nb_pesq_val]
        else:
            ...
    except Exception as e:
        ...
        # warnings.warn(str(e))
    return [None, None]


def recover_scale(preds: Tensor, mixture: Tensor, scale_src_together: bool, norm_if_exceed_1: bool = True) -> Tensor:
    """recover wav's original scale by solving min ||Y^T a - X||F, cuz sisdr will lose scale

    Args:
        preds: prediction, shape [batch, n_src, time]
        mixture: mixture or noisy or reverberant signal, shape [batch, time]
        scale_src_together: keep the relative ennergy level between sources. can be used for scale-invariant SA-SDR
        norm_max_if_exceed_1: norm the magitude if exceeds one

    Returns:
        Tensor: the scale-recovered preds
    """
    # TODO: add some kind of weighting mechanism to make the predicted scales more precise
    # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
    if scale_src_together:
        a = torch.linalg.lstsq(preds.sum(dim=-2, keepdim=True).transpose(-1, -2), mixture.unsqueeze(-1)).solution
    else:
        a = torch.linalg.lstsq(preds.transpose(-1, -2), mixture.unsqueeze(-1)).solution

    preds = preds * a

    if norm_if_exceed_1:
        # normalize the audios so that the maximum doesn't exceed 1
        max_vals = torch.max(torch.abs(preds), dim=-1).values
        norm = torch.where(max_vals > 1, max_vals, 1)
        preds = preds / norm.unsqueeze(-1)
    return preds

import traceback
from models.utils.asr_metric import Whisper
from pypesq import pesq as pypesq

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI']

# import speechmetrics 

class MOS(torch.nn.Module):
    def __init__(self, fs = 8000, ): 
        super(MOS, self).__init__()
        self.sm = speechmetrics.load("mosnet", window = 10)
        self.fs = fs
    
    def calculate(self, x, fs):
        return torch.from_numpy(self.sm(x, rate = fs)["mosnet"])
    
    def forward(self, preds, fs):
        # preds: spk x time 
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        
        preds = preds.detach().cpu().numpy()
        
        assert preds.ndim == 2, "preds should be 2d array - spk x time"

        return torch.cat([self.calculate(preds[i], fs = fs) for i in range(preds.shape[0])])


# speechmetrics_mos = MOS()

def construct_audio_MetricCollection(
    metrics: List[str],
    prefix: str = '',
    postfix: str = '',
    fs: int = None,
) -> MetricCollection:
    md: Dict[str, Metric] = {}
    for m in metrics:
        mname = prefix + m.lower() + postfix
        if m.upper() == 'SDR':
            md[mname] = SignalDistortionRatio()
        elif m.upper() == 'SI_SDR':
            md[mname] = ScaleInvariantSignalDistortionRatio()
        elif m.upper() == 'SI_SNR':
            md[mname] = ScaleInvariantSignalNoiseRatio()
        elif m.upper() == 'SNR':
            md[mname] = SignalNoiseRatio()
        elif m.upper() == 'NB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='nb').cpu()
        elif m.upper() == 'WB_PESQ':
            md[mname] = PerceptualEvaluationSpeechQuality(fs, mode='wb').cpu()
        elif m.upper() == 'STOI':
            md[mname] = ShortTimeObjectiveIntelligibility(fs).cpu()
        else:
            raise ValueError('Unkown audio metric ' + m)

    return MetricCollection(md)


def cal_metrics(
    preds: Tensor,
    target: Tensor,
    original: Union[Tensor, Dict[str, Tensor]],
    mc: MetricCollection,
    input_mc: Optional[MetricCollection] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """calculate metrics, input_metrics, imp_metrics

    Args:
        preds: prediction
        target: target
        original: original signal or input_metrics
        mc: MetricCollection 
        input_mc: Input MetricCollection if original signal is given, else None

    Returns:
        metrics, input_metrics, imp_metrics
    """
    metrics = mc(preds, target)
    if isinstance(original, Tensor):
        if input_mc == None:
            raise ValueError('input_mc cannot be None when original signal is given, i.e. original is a Tensor')
        input_metrics = input_mc(original, target)
    else:
        input_metrics = original
    imp_metrics = {}
    for k, v in metrics.items():
        v = v.detach().cpu()
        iv = input_metrics['input_' + k].detach().cpu()
        metrics[k] = v
        input_metrics['input_' + k] = iv
        imp_metrics[k + '_i'] = v - iv

    return metrics, input_metrics, imp_metrics


def get_metric_list_on_device(device: Optional[str]):
    metric_device = {
        None: ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ', 'STOI', 'PYPESQ', 'ESTOI', "WER", "MOS"],
        "cpu": ['NB_PESQ', 'WB_PESQ', 'PYPESQ', 'STOI', 'ESTOI'],
        "gpu": ['SDR', 'SI_SDR', 'SNR', 'SI_SNR', "WER", "MOS"],
    }
    return metric_device[device]


def cal_metrics_functional(
        metric_list: List[str],
        preds: Tensor,
        target: Tensor,
        original: Optional[Tensor],
        fs: int,
        device_only: Literal['cpu', 'gpu', None] = None,  # cpu-only: pesq, stoi;
        asr = None, 
        target_transcript: str = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    if device_only is None or device_only == 'cpu':
        preds_cpu = preds.detach().cpu()
        target_cpu = target.detach().cpu()
        original_cpu = original.detach().cpu() if original is not None else None
    else:
        preds_cpu = None
        target_cpu = None
        original_cpu = None

    input_metrics = {}
    metrics = {}
    imp_metrics = {}


    for m in metric_list: 
        try:
            mname = m.lower()
            if m.upper() not in get_metric_list_on_device(device=device_only):
                continue

            if m.upper() == 'SDR':
                ## not use signal_distortion_ratio for it gives NaN sometimes
                metric_func = lambda: signal_distortion_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: signal_distortion_ratio(original, target).detach().cpu()
                # assert preds.dim() == 2 and target.dim() == 2 and original.dim() == 2, '(spk, time)!'
                # metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), preds_cpu.numpy(), False)[0]).mean().detach().cpu()
                # input_metric_func = lambda: torch.tensor(bss_eval_sources(target_cpu.numpy(), original_cpu.numpy(), False)[0]).mean().detach().cpu()
            elif m.upper() == 'SI_SDR':
                metric_func = lambda: scale_invariant_signal_distortion_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: scale_invariant_signal_distortion_ratio(original, target).detach().cpu()
            elif m.upper() == 'MOS':
                metric_func = lambda: speechmetrics_mos(preds, fs = fs)
                input_metric_func = lambda: speechmetrics_mos(original, fs = fs)          
            elif m.upper() == 'SI_SNR':
                metric_func = lambda: scale_invariant_signal_noise_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: scale_invariant_signal_noise_ratio(original, target).detach().cpu()
            elif m.upper() == 'SNR':
                metric_func = lambda: signal_noise_ratio(preds, target).detach().cpu()
                input_metric_func = lambda: signal_noise_ratio(original, target).detach().cpu()
            elif m.upper() == 'WER':
                metric_func = lambda: asr(preds, target_transcript, return_transcript = True)
                input_metric_func = lambda: asr(original, target_transcript, return_transcript = True)
            elif m.upper() == 'NB_PESQ':
                metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'nb', n_processes=0)
                input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'nb', n_processes=0)
            
            elif m.upper() == 'PYPESQ':
                metric_func = lambda: torch.Tensor([pypesq(ref = target_cpu.reshape(-1).numpy(), deg = preds_cpu.reshape(-1).numpy(), fs = fs)])
                input_metric_func = lambda: torch.Tensor([pypesq(ref = target_cpu.reshape(-1).numpy(), deg = original_cpu.reshape(-1).numpy(), fs = fs)])
            elif m.upper() == 'WB_PESQ':
                metric_func = lambda: perceptual_evaluation_speech_quality(preds_cpu, target_cpu, fs, 'wb', n_processes=0)
                input_metric_func = lambda: perceptual_evaluation_speech_quality(original_cpu, target_cpu, fs, 'wb', n_processes=0)
            elif m.upper() == 'STOI':
                metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs)
                input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs)
            elif m.upper() == 'ESTOI':
                metric_func = lambda: short_time_objective_intelligibility(preds_cpu, target_cpu, fs, extended=True)
                input_metric_func = lambda: short_time_objective_intelligibility(original_cpu, target_cpu, fs, extended=True)
            else:
                raise ValueError('Unkown audio metric ' + m)

            if m.upper() == 'WB_PESQ' and fs == 8000:
                warnings.warn("There is narrow band (nb) mode only when sampling rate is 8000Hz")
                continue  # Note there is narrow band (nb) mode only when sampling rate is 8000Hz

            if True:
                

                
                if m.upper() == 'WER':
                    transcription, wer = metric_func()
                    m_val = wer.cpu().numpy() 
                    metrics[mname + "_transcription"] = transcription
                else:
                    m_val = metric_func().cpu().numpy()
                
                
                metrics[mname] = np.mean(m_val).item()
                metrics[mname + '_all'] = m_val.tolist()  # _all means not averaged
                
                if original is None:
                    continue
                
                if 'input_' + mname not in input_metrics.keys():
                    if m.upper() == 'WER':
                        transcription, im_val = input_metric_func() 
                        im_val = im_val.cpu().numpy()
                        input_metrics['input_' + mname  + "_transcription"] = transcription
                    
                    else:
                        im_val = input_metric_func().cpu().numpy() if original is not None else np.nan
                    
                    input_metrics['input_' + mname] = np.mean(im_val).item()
                    input_metrics['input_' + mname + '_all'] = im_val.tolist()

                imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]  # _i means improvement
                imp_metrics[mname + '_all' + '_i'] = (m_val - im_val).tolist()
        except Exception as e: 
            # traceback.print_exception(type(e), e, e.__traceback__)

            warnings.warn(f"Error: {mname} - {e}")
            metrics[mname] = None
            metrics[mname + '_all'] = None
            if 'input_' + mname not in input_metrics.keys():
                input_metrics['input_' + mname] = None
                input_metrics['input_' + mname + '_all'] = None
            imp_metrics[mname + '_i'] = None
            imp_metrics[mname + '_i' + '_all'] = None

    return metrics, input_metrics, imp_metrics


def mypesq(preds: np.ndarray, target: np.ndarray, mode: str, fs: int) -> np.ndarray:
    # 使用ndarray是因为tensor会在linux上会导致一些多进程的错误
    ori_shape = preds.shape
    if type(preds) == Tensor:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    else:
        assert type(preds) == np.ndarray, type(preds)
        assert type(target) == np.ndarray, type(target)

    if preds.ndim == 1:
        pesq_val = pesq_backend.pesq(fs=fs, ref=target, deg=preds, mode=mode)
    else:
        preds = preds.reshape(-1, ori_shape[-1])
        target = target.reshape(-1, ori_shape[-1])
        pesq_val = np.empty(shape=(preds.shape[0]))
        for b in range(preds.shape[0]):
            pesq_val[b] = pesq_backend.pesq(fs=fs, ref=target[b, :], deg=preds[b, :], mode=mode)
        pesq_val = pesq_val.reshape(ori_shape[:-1])
    return pesq_val


def cal_pesq(ys: np.ndarray, ys_hat: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    try:
        if sample_rate == 16000:
            wb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='wb').mean()
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [wb_pesq_val, nb_pesq_val]
        elif sample_rate == 8000:
            nb_pesq_val = mypesq(preds=ys_hat, target=ys, fs=sample_rate, mode='nb').mean()
            return [None, nb_pesq_val]
        else:
            ...
    except Exception as e:
        ...
        # warnings.warn(str(e))
    return [None, None]


def recover_scale(preds: Tensor, mixture: Tensor, scale_src_together: bool, norm_if_exceed_1: bool = True) -> Tensor:
    """recover wav's original scale by solving min ||Y^T a - X||F, cuz sisdr will lose scale

    Args:
        preds: prediction, shape [batch, n_src, time]
        mixture: mixture or noisy or reverberant signal, shape [batch, time]
        scale_src_together: keep the relative ennergy level between sources. can be used for scale-invariant SA-SDR
        norm_max_if_exceed_1: norm the magitude if exceeds one

    Returns:
        Tensor: the scale-recovered preds
    """
    # TODO: add some kind of weighting mechanism to make the predicted scales more precise
    # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
    if scale_src_together:
        a = torch.linalg.lstsq(preds.sum(dim=-2, keepdim=True).transpose(-1, -2), mixture.unsqueeze(-1)).solution
    else:
        a = torch.linalg.lstsq(preds.transpose(-1, -2), mixture.unsqueeze(-1)).solution

    preds = preds * a

    if norm_if_exceed_1:
        # normalize the audios so that the maximum doesn't exceed 1
        max_vals = torch.max(torch.abs(preds), dim=-1).values
        norm = torch.where(max_vals > 1, max_vals, 1)
        preds = preds / norm.unsqueeze(-1)
    return preds
