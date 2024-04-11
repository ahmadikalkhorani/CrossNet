from os import stat
import torch
from typing import Optional
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np 



class LearningRateScheduler(LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, ):
        pass
        
     
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for g in optimizer.param_groups:
            return g['lr']

class WarmupLRScheduler(LearningRateScheduler):
    def __init__(
            self,
            update_steps: int,
            warmup_steps: int,
            peak_lr: float,
            init_lr: float,
            **kwargs
    ) -> None:
        super(WarmupLRScheduler, self).__init__()
        
        args = locals().copy()  # capture the parameters passed to this function or their edited values
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
            
        self.warmup_rate = None
    
 

    def step(self, optimizer, val_loss: Optional[torch.FloatTensor] = None):
        if self.warmup_rate is None:
            if self.warmup_steps != 0:
                warmup_rate = self.peak_lr - self.init_lr
                self.warmup_rate = warmup_rate / self.warmup_steps
            else:
                self.warmup_rate = 0
            
        
        
        
        if self.update_steps <= self.warmup_steps:
            # lr = self.init_lr + self.warmup_rate * self.update_steps
            lr = self.peak_lr -0.5 * (self.peak_lr - self.init_lr) * (1 + torch.cos(torch.tensor([torch.pi * self.update_steps/self.warmup_steps])))
            lr = lr.item()
            self.set_lr(optimizer, lr)
            self.lr = lr
  
        
        self.update_steps += 1
        return self.lr


class MyReduceLROnPlateau(LearningRateScheduler):
    def __init__(
            self,
            mode='min', factor=0.9, patience=3,
            threshold=5e-2, threshold_mode='abs', cooldown=0,
            min_lr=0, eps=1e-8, verbose=True, *args, **kwargs,
    ) -> None:
        super(MyReduceLROnPlateau, self).__init__()
        
        args = locals().copy()  # capture the parameters passed to this function or their edited values
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
        
        self.best_loss = None
        self.bad_epochs = 0
        self.update_steps = -1   
    
    def improved(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True
        else:
            if (val_loss < self.best_loss)&(torch.abs(val_loss - self.best_loss) >= self.threshold):
                self.best_loss = val_loss
                self.bad_epochs = 0
                return True
            else:
                self.bad_epochs += 1
                return False
        
    def verbose(self, optimizer, val_loss):
        N = 1 + int(np.ceil(np.abs(np.log(self.threshold))))
        if self.verbose:
            print(f'MyReduceLROnPlateau => Epoch {self.update_steps} | lr: {self.get_lr(optimizer)} | val_loss: {round(val_loss.item(), N)} | best_loss: {round(self.best_loss.item(), N)} | bad_epochs: {self.bad_epochs}/{self.patience}')
    
    def step(self, optimizer: torch.optim.Optimizer, val_loss: torch.FloatTensor):
        self.update_steps += 1
        
        lr = self.get_lr(optimizer)
        
        if not self.improved(val_loss):
            if self.bad_epochs >= self.patience:
                lr = lr * self.factor
                self.bad_epochs = 0
    
            
        
        self.set_lr(optimizer, lr)
        
        self.verbose(optimizer, val_loss)
        
        self.lr = lr

        
        return self.lr
    
    



class ReduceLROnPlateau(LearningRateScheduler):
    def __init__(
            self,
            mode='min', factor=0.9, patience=3,
            threshold=5e-2, threshold_mode='abs', cooldown=0,
            min_lr=0, eps=1e-8, verbose=True, *args, **kwargs,
    ) -> None:
        super(ReduceLROnPlateau, self).__init__()
        
        args = locals().copy()  # capture the parameters passed to this function or their edited values
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
        
        self.best_loss = None
        self.bad_epochs = 0
        self.update_steps = -1   
    
    def improved(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True
        else:
            if (val_loss < self.best_loss)&(torch.abs(val_loss - self.best_loss) >= self.threshold):
                self.best_loss = val_loss
                self.bad_epochs = 0
                return True
            else:
                self.bad_epochs += 1
                return False
        
    def verbose(self, optimizer, val_loss):
        N = 1 + int(np.ceil(np.abs(np.log(self.threshold))))
        if self.verbose:
            print(f'ReduceLROnPlateau => Epoch {self.update_steps} | lr: {self.get_lr(optimizer)} | val_loss: {round(val_loss.item(), N)} | best_loss: {round(self.best_loss.item(), N)} | bad_epochs: {self.bad_epochs}/{self.patience}')
    
    def step(self, optimizer, val_loss: torch.FloatTensor):
        self.update_steps += 1
        
        lr = self.get_lr(optimizer)
        
        if not self.improved(val_loss):
            if self.bad_epochs >= self.patience:
                lr = lr * self.factor
                self.bad_epochs = 0
    
            
        
        self.set_lr(optimizer, lr)
        
        self.verbose(optimizer, val_loss)
        
        self.lr = lr

        
        return self.lr
    
    

class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            mode='min', factor=0.1, patience=3,
            threshold=5e-2, threshold_mode='abs', cooldown=0,
            min_lr=0, eps=1e-8, verbose=True, 
            
            warmup_steps: int = 5,
            peak_lr: float = 1E-4,
            init_lr: float = 1E-6, 
            update_steps: int = 0,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__()
        
        args = locals().copy()  # capture the parameters passed to this function or their edited values
        print(args)
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
            
            
        
        self.schedulers = [
            WarmupLRScheduler(
                update_steps = update_steps,
                warmup_steps = warmup_steps,
                peak_lr = peak_lr,
                init_lr = self.get_lr(optimizer),
            ),
            MyReduceLROnPlateau(
                mode=mode, factor=factor, patience=patience,
                threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                min_lr=min_lr, eps=eps, verbose=verbose,
            ),
        ]
        
        self.optimizer = optimizer
        
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        state_dct = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        for scheduler in self.schedulers:
            name = scheduler.__class__.__name__
            state_dct[name] = scheduler.state_dict()
        return state_dct

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for scheduler in self.schedulers:
            name = scheduler.__class__.__name__
            if name in state_dict.keys():
                scheduler.load_state_dict(state_dict.pop(name))
            
    
        self.__dict__.update(state_dict)
     

    def _decide_stage(self):
        if self.update_steps <= self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        print("val_loss: ", val_loss, "stage: ", stage, "self.warmup_steps: ", self.warmup_steps, "self.update_steps: ", self.update_steps)

        self.schedulers[stage].step(optimizer = self.optimizer, val_loss = val_loss)
       

        self.update_steps += 1

        return self.get_lr(self.optimizer)
    

