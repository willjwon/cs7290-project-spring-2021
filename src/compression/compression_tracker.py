import torch
import struct
from abc import ABC, abstractmethod


class CompressionTracker(ABC):
    def __init__(self):
        self.bins_count = 3000
        self.compression_ratio_histogram = [0 for _ in range(self.bins_count)]

    @staticmethod
    def pack_tensor_in_bytes(data: torch.Tensor) -> bytes:
        data_list_format = data.clone().tolist()
        return struct.pack('d' * len(data_list_format), *data_list_format)

    def record_compression_ratio(self, orig: bytes, compressed: bytes) -> None:
        ratio = len(compressed) / len(orig)
        bin = int(ratio * 1000)

        if 0 <= bin < self.bins_count:
            self.compression_ratio_histogram[int(ratio * 1000)] += 1
        # print(f"before: {len(orig)}, after: {len(compressed)}, ratio: {ratio}")

    @abstractmethod
    def process_message(self, message: torch.Tensor) -> None:
        pass
