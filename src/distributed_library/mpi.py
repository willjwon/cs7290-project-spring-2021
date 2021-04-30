from typing import Dict
from src.compression.compression_tracker import CompressionTracker
import torch


def reduce_scatter(nodes_count: int, message: Dict[int, Dict[int, torch.Tensor]], tracker: CompressionTracker):
    for step in range(1, nodes_count):
        for src_node in range(nodes_count):
            dest_node = (src_node + 1) % nodes_count
            packet_id = (src_node - step) % nodes_count

            tracker.process_message(message[src_node][packet_id])
            message[dest_node][packet_id] += message[src_node][packet_id]


def all_gather(nodes_count: int, message: Dict[int, Dict[int, torch.Tensor]], tracker: CompressionTracker):
    for step in range(nodes_count - 1):
        for src_node in range(nodes_count):
            dest_node = (src_node + 1) % nodes_count
            packet_id = (src_node - step) % nodes_count

            tracker.process_message(message[src_node][packet_id])
            message[dest_node][packet_id] = message[src_node][packet_id]


def all_reduce(nodes_count: int, message: Dict[int, Dict[int, torch.Tensor]], tracker: CompressionTracker):
    reduce_scatter(nodes_count=nodes_count, message=message, tracker=tracker)
    all_gather(nodes_count=nodes_count, message=message, tracker=tracker)
