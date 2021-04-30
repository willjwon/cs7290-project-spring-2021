import torch
import deflate
from .compression_tracker import CompressionTracker


class CompressionTrackerDeflate(CompressionTracker):
    def __init__(self):
        super(CompressionTrackerDeflate, self).__init__()

    @staticmethod
    def compress(data: bytes) -> bytes:
        return deflate.gzip_compress(data, 1)

    def process_message(self, message: torch.Tensor) -> None:
        original_message = CompressionTrackerDeflate.pack_tensor_in_bytes(data=message)
        compressed_message = CompressionTrackerDeflate.compress(data=original_message)

        self.record_compression_ratio(orig=original_message, compressed=compressed_message)


class CompressionTrackerDeflateConcat(CompressionTracker):
    def __init__(self):
        super(CompressionTrackerDeflateConcat, self).__init__()
        self.message_pool = bytes()
        self.chunk_size = 64 << 10  # 64kB

    @staticmethod
    def compress(data: bytes) -> bytes:
        return deflate.gzip_compress(data, 1)

    def process_message(self, message: torch.Tensor) -> None:
        new_message = CompressionTrackerDeflateConcat.pack_tensor_in_bytes(data=message)
        self.message_pool += new_message

        while len(self.message_pool) >= self.chunk_size:
            # 64kB chunk
            original_message = self.message_pool[:self.chunk_size]
            self.message_pool = self.message_pool[self.chunk_size:]
            compressed_message = CompressionTrackerDeflateConcat.compress(data=original_message)

            self.record_compression_ratio(orig=original_message, compressed=compressed_message)
