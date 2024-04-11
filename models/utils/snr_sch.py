
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from lightning_utilities.core.rank_zero import rank_prefixed_message
from torch import Tensor

import pytorch_lightning as pl
from lightning_lite.utilities.rank_zero import _get_rank
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)



class SNRLogger(Callback):
    def __init__(self, logging_interval: str = "epoch"):
        if logging_interval not in (None, 'step', 'epoch'):
            raise MisconfigurationException(
                'logging_interval should be `step` or `epoch` or `None`.'
            )

        self.logging_interval = logging_interval
        self.lrs = None
        self.lr_sch_names = []

    def on_train_start(self, trainer, pl_module):
        """ Called before training, determines unique names for all lr
            schedulers in the case of multiple of the same type or in
            the case of multiple parameter groups
        """
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use LearningRateLogger callback with Trainer that has no logger.'
            )


    def on_train_batch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if self.logging_interval != 'epoch':
            interval = 'step' if self.logging_interval is None else 'any'
            latest_stat = self._extract_snr(trainer, )

            if latest_stat:
                for logger in trainer.loggers:
                    logger.log_metrics(latest_stat, step=trainer.fit_loop.epoch_loop._batches_that_stepped)


    def on_train_epoch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if self.logging_interval != 'step':
            latest_stat = self._extract_snr(trainer)
            print(latest_stat)
            
            if trainer.logger is None:
                print("SNRLogger: Trainer logger is None")
                
            if latest_stat:
                for logger in trainer.loggers:
                    logger.log_metrics(latest_stat, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def _extract_snr(self, trainer):
        latest_stat = {
            "SNR_lb": trainer.datamodule.train.noise_scale[0],
            "SNR_ub": trainer.datamodule.train.noise_scale[1],
        }

        return latest_stat




class SNRSheduler(Callback):

    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
        improve_snr_bounds: float = 5.0, # dB
        initial_noise_scale: Tuple[float, float] = [-30, -20],
        final_noise_scale: Tuple[float, float] = [-10, 10],
    ):
        super().__init__()
        self.snr_min, self.snr_max = final_noise_scale
        self.snr_bounds = initial_noise_scale
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = -1
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only
        
        self.improve_snr_bounds = improve_snr_bounds

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"SNR scheduler conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `SNRScheduler` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
            "snr_bounds": self.snr_bounds
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        self.snr_bounds = state_dict["snr_bounds"]



        print("Loaded State Dict for SNR Callbakc: \n", self.snr_bounds)
        # try:
        #     self.snr_bounds = state_dict["snr_bounds"]
        # except Exception as err:
        #     print(err)
        #     self.snr_bounds = (self.snr_min, self.snr_max)

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking
    
    def on_train_start(self, trainer, pl_module):
      
        if self.snr_bounds is not None:
            trainer.datamodule.train.change_noise_scale(self.snr_bounds)
            print("noise_scale : ", trainer.datamodule.train.noise_scale)
        

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_changing_snr_check(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_changing_snr_check(trainer, pl_module)

    def _run_changing_snr_check(self, trainer: "pl.Trainer", pl_module : "pl.LightningModule") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_change, reason = self._evaluate_changing_snr_criteria(current)
        if self.snr_bounds is None:
            noise_bounds = trainer.datamodule.train.noise_db_bounds
            self.snr_bounds = noise_bounds
        else:
            noise_bounds = self.snr_bounds
        new_noise_bounds = noise_bounds
        if should_change:
            new_noise_bounds = (min(self.snr_min, new_noise_bounds[0] + self.improve_snr_bounds), min(new_noise_bounds[1] + self.improve_snr_bounds, self.snr_max))
            # new_noise_bounds = (new_noise_bounds[0], min(new_noise_bounds[1] + self.improve_snr_bounds, self.snr_max))
            # pl_module.datamodule.train_dataset.noise_db_bounds = new_noise_bounds 
            
            trainer.datamodule.train.change_noise_scale(new_noise_bounds)
            
            self.snr_bounds = new_noise_bounds
        
            print(f"Changing SNR bounds to: ", new_noise_bounds)
        else:
            print(f"SNR bounds: ", new_noise_bounds)
                        

        # stop every ddp process if any world process decides to stop
        # should_change = trainer.strategy.reduce_boolean_decision(should_change, all=False)
        # trainer.should_change = trainer.should_change or should_change
        # if should_change:
        #     self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason + '\n' + f'Chaning to: {new_noise_bounds}', self.log_rank_zero_only)

    def _evaluate_changing_snr_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_change = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_change = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to Change SNR bounds."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_change = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to Change SNR bounds."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_change = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to Change SNR bounds."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_change = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_change = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to Change SNR bounds."
                )
                self.wait_count = -1
                
        print("Reason: ", reason)
        
        return should_change, reason

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)














class AVFreezeSheduler(Callback):

    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
        **kwargs
    ):
        super().__init__()
 
        self.snr_bounds = None
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only
        

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        
        self.epoch = 0

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
            "epoch": self.epoch
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        self.epoch = state_dict["epoch"]



        print("Loaded State Dict for SNR Callbakc: \n", self.snr_bounds)
        # try:
        #     self.snr_bounds = state_dict["snr_bounds"]
        # except Exception as err:
        #     print(err)
        #     self.snr_bounds = (self.snr_min, self.snr_max)

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_changing_snr_check(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_changing_snr_check(trainer, pl_module)

    def _run_changing_snr_check(self, trainer: "pl.Trainer", pl_module : "pl.LightningModule") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_change, reason = self._evaluate_changing_snr_criteria(current)
        if self.snr_bounds is None:
            noise_bounds = pl_module.datamodule.train_dataset.noise_db_bounds
            self.snr_bounds = noise_bounds
        else:
            noise_bounds = self.snr_bounds
        new_noise_bounds = noise_bounds
        if should_change:
            # TODO: probably need to increase the `upper` bound only
            new_noise_bounds = (min(self.snr_min, new_noise_bounds[0] + self.improve_snr_bounds), min(new_noise_bounds[1] + self.improve_snr_bounds, self.snr_max))
            # new_noise_bounds = (new_noise_bounds[0], min(new_noise_bounds[1] + self.improve_snr_bounds, self.snr_max))
            pl_module.datamodule.train_dataset.noise_db_bounds = new_noise_bounds 
            
            self.snr_bounds = new_noise_bounds
        
            print(f"Changing SNR bounds to: ", new_noise_bounds)
        else:
            print(f"SNR bounds: ", new_noise_bounds)
                        

        # stop every ddp process if any world process decides to stop
        # should_change = trainer.strategy.reduce_boolean_decision(should_change, all=False)
        # trainer.should_change = trainer.should_change or should_change
        # if should_change:
        #     self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason + '\n' + f'Chaning to: {new_noise_bounds}', self.log_rank_zero_only)

    def _evaluate_changing_snr_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_change = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_change = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to Change SNR bounds."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_change = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to Change SNR bounds."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_change = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to Change SNR bounds."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_change = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_change = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to Change SNR bounds."
                )
                
                self.wait_count = 0
        print("Reason: ", reason)
        
        return should_change, reason

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)