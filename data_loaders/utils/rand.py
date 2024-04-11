import torch
from typing import List, Any

def randint(g: torch.Generator, low: int, high: int) -> int:
    """return a value sampled in [low, high)
    """
    if low == high:
        return low
    r = torch.randint(low=low, high=high, size=(1,), generator=g, device='cpu')
    return r[0].item()  # type:ignore


def randfloat(g: torch.Generator, low: float, high: float) -> float:
    """return a value sampled in [low, high)
    """
    if low == high:
        return low
    r = torch.rand(size=(1,), generator=g, device='cpu')[0].item()
    return float(low + r * (high - low))

def randnormal(g: torch.Generator, mean: float, std: float) -> Any:
    return torch.normal(mean, std, size=(1, )).item()

def randchoice(g: torch.Generator, seq: List[Any]) -> Any:
    return seq[ randint(g, 0, len(seq)) ]
