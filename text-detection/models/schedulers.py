import math
import warnings
from torch.optim.lr_scheduler import LRScheduler


class PretrainingScheduler(LRScheduler):
    """
    Cosine learning rate scheduler optimized for pretraining, with:
    - Linear warmup
    - Cosine decay
    - Optional restart cycles
    - Min learning rate threshold

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps (250000)
        warmup_steps: Number of warmup steps (default: 2% of total_steps)
        min_lr_ratio: Minimum learning rate as a fraction of max lr (default: 0.1)
        cycles: Number of cosine cycles (default: 1)
    """

    def __init__(
        self,
        optimizer,
        total_steps=250000,
        warmup_steps=0,
        eta_min=0,
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        step = self.last_epoch

        # Linear warmup
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]

        # Cosine decay
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.total_steps))
            / 2
            for base_lr in self.base_lrs
        ]
