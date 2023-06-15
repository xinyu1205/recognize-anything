from typing import Any, Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def inference(
    model: Callable,
    datalist: List[Any],
    data_processor: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    output_processor: Optional[Callable] = None,
    to_cpu: bool = True,
):
    """Perform inference with the given model."""

    loader = _get_loader(datalist, data_processor, batch_size, num_workers)
    n = len(loader.dataset)
    outputs = None
    device = None

    pos = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Inference"):

            out = model(data)

            if output_processor is not None:
                out = output_processor(out)
            if device is None:
                device = out.device
            if to_cpu:
                out = out.cpu()
            if outputs is None:
                outputs = torch.empty(
                    n, *out.shape[1:], device=device, dtype=out.dtype)
            step = out.shape[0]
            outputs[pos:pos+step] = out
            pos += step

    return outputs


def _get_loader(
    datalist: List[Any],
    processor: Optional[Callable],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:

    class _Dataset(Dataset):
        def __init__(self, data, processor):
            self._data = data
            self._processor = processor
            self._length = len(data)

        def __len__(self):
            return self._length

        def __getitem__(self, index):
            out = self._data[index]
            if self._processor:
                out = self._processor(out)
            return out

    return DataLoader(
        _Dataset(datalist, processor),
        batch_size=32 if batch_size is None else batch_size,
        num_workers=6 if num_workers is None else num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
