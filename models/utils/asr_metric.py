
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from whisper.normalizers import EnglishTextNormalizer
from torchmetrics.text import WordErrorRate
import re



class EasyComTextNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.replacements = {
            "dunno": "do not know", "okay": "ok",  "aw ": "0 ", "yep": "yeah", "-eah": "yeah", "plenty full": "plentyfull",
            "nope": "no", " ah. ": " ","oh. ": " ", "alright": "all right", "nah": "no", "woah": "whoa", " Kay.": " ok", "-Kay.": "ok",
            " bout ": " about ", "-owntown":"downtown", "y'know": "you know",
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            }
        self.whisper_normalizer = EnglishTextNormalizer()
    def forward(self, txt):
        txt = txt.lower()
        txt = re.sub(r'\[.*?\]', '', txt).strip() # remove characters between []
        txt = re.sub(r'{.*?}', '', txt).strip() # remove characters between {}
        for k, v in self.replacements.items():
            txt = txt.replace(k, v)
        txt = re.sub(r'[\w\']*-', '', txt).replace("  ", " ").strip() # remove words ending with -
        txt = self.whisper_normalizer(txt)
        return txt




class Wav2Vec2(torch.nn.Module):
    def __init__(self, sr, device="cuda:0") -> None:
        super().__init__()
        # load model and processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        model.config.forced_decoder_ids = None
        
        self.restorer = PunctuationRestorer.from_pretrained("openai/whisper-base.en")
        self.restorer.to(device)      

        self.sr = sr
        self.device = device
        self.processor = processor 
        self.model = model 
    
    def name(self):
        return "Wav2Vec2"
    def restore_punc(self, x, transcript):
        res = self.restorer(x.flatten(), transcript.strip(), sampling_rate=self.sr, num_beams=1)
        
        if isinstance(res, tuple):
            return res[0]
        return res
        
    def forward(self, s):
        s = torchaudio.functional.resample(s.reshape(-1), orig_freq=self.sr, new_freq=16000).to(self.device)
        input_features = self.processor(s, sampling_rate=16000, return_tensors="pt").input_values 
        logits = self.model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        # restored_text, log_probs = restorer(s, transcription[0].strip(), sampling_rate=16000, num_beams=1)
        
        return transcription

class Whisper(torch.nn.Module):
    def __init__(self, sr, device="cuda:0", model_name = "large", reduce_wer_func: str = "min") -> None:
        super().__init__()
        assert model_name in ["tiny", "base", "small", "medium", "large"]
        if model_name != "large":
            model_name += ".en"
        # load model and processor
        processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_name}")
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_name}")
        model.config.forced_decoder_ids = None
        
        
        self.restorer = None

        self.sr = sr
        self.model_name = model_name
        self.device = device
        self.processor = processor 
        self.model = model # .to(device)
        self.text_normalizer = EasyComTextNormalizer()
        self.WER = WordErrorRate()
        
        self.reduce_wer_func = {"min": min, "max": max, "mean": np.mean}[reduce_wer_func]

        
    def name(self):
        return "Whisper" + f"_{self.model_name}"
    
    def restore_punc(self, x, transcript):
        if self.restorer is None:
            self.restorer = PunctuationRestorer.from_pretrained(f"openai/whisper-base.en")
            self.restorer.to(self.device)   
        
        res = self.restorer(x.flatten(), transcript.strip(), sampling_rate=self.sr, num_beams=1)
        
        if isinstance(res, tuple):
            return res[0]
        return res
    
    def pit_wer(self, metric_func, preds, target, reduce_wer_func = min):
        # preds : list with len 2
        # target : list with len 2

        wer1 = metric_func(preds, target)
        
        preds.reverse()
        wer2 = metric_func(preds, target)
        
        if wer1 <= wer2:
            
            preds.reverse() # back to the original order
        
        wers = [metric_func(preds = preds[i], target = target[i]).item() for i in range(len(preds))]
        
        return reduce_wer_func(wers), preds
        
    def forward(self, s: torch.Tensor, target_transcription: list = None, return_transcript: bool = True, return_wer: bool = True):
        # s is batched signal [spk, T]
        if isinstance(target_transcription, str):
            target_transcription = [target_transcription]
        
        s = s.squeeze()
        
        assert s.ndim <= 2
        
        if s.ndim == 1:
            s = s.reshape(1, -1)
        
        # normalize the input signal 
        s = (s - s.mean()) / s.std()
        
        
        preds = [self.forward_single(s[b]) for b in range(s.shape[0])]
            
        if target_transcription is None:
            return preds
        
        preds = [self.text_normalizer(txt) for txt in preds]
        target = [self.text_normalizer(txt) for txt in target_transcription]
        
        if not return_wer:
            return preds
        
        wer, preds = self.pit_wer(metric_func = self.WER, preds = preds, target = target, reduce_wer_func = self.reduce_wer_func)
        
        wer = torch.tensor([wer])
        
        if return_transcript:
            return preds, wer
        return wer
        

    def forward_single(self, s: torch.Tensor):
        s = torchaudio.functional.resample(s.reshape(-1), orig_freq=self.sr, new_freq=16000)
        input_features = self.processor(s.detach().cpu(), sampling_rate=16000, return_tensors="pt").input_features  
        predicted_ids = self.model.generate(input_features.to(self.model.device)) 
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # restored_text, log_probs = restorer(s, transcription[0].strip(), sampling_rate=16000, num_beams=1)
        return transcription[0]
        
        
