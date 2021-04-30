import torch
import lz4.frame
from .compression_tracker import CompressionTracker


class CompressionTrackerLz4(CompressionTracker):
    def __init__(self):
        super(CompressionTrackerLz4, self).__init__()

    @staticmethod
    def compress(data: bytes) -> bytes:
        return lz4.frame.compress(data=data)

    def process_message(self, message: torch.Tensor) -> None:
        original_message = CompressionTrackerLz4.pack_tensor_in_bytes(data=message)
        compressed_message = CompressionTrackerLz4.compress(data=original_message)

        self.record_compression_ratio(orig=original_message, compressed=compressed_message)


class CompressionTrackerLz4Concat(CompressionTracker):
    def __init__(self):
        super(CompressionTrackerLz4Concat, self).__init__()
        self.message_pool = bytes()
        self.chunk_size = 64 << 10  # 64kB

    @staticmethod
    def compress(data: bytes) -> bytes:
        return lz4.frame.compress(data=data)

    def process_message(self, message: torch.Tensor) -> None:
        new_message = CompressionTrackerLz4Concat.pack_tensor_in_bytes(data=message)
        self.message_pool += new_message

        while len(self.message_pool) >= self.chunk_size:
            # 64kB chunk
            original_message = self.message_pool[:self.chunk_size]
            self.message_pool = self.message_pool[self.chunk_size:]
            compressed_message = CompressionTrackerLz4Concat.compress(data=original_message)

            self.record_compression_ratio(orig=original_message, compressed=compressed_message)
