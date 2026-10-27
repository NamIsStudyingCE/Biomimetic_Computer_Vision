"""
PyTorch Data Loader & Utility Module for FlyWire Connectome Graph
Biomimetic Computer Vision Project
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

DEFAULT_GRAPH_PATH = Path(r"D:\Biomimetic Computer Vision\data\flywire\flywire_visual_graph.pt")

class FlyWireConnectomeLoader:
    """
    High-performance frozen biological graph loader.
    Provides synaptic weight matrices, layer partition indices, and 2D spatial convolution kernels.
    """
    def __init__(self, graph_path: Path = DEFAULT_GRAPH_PATH):
        if not graph_path.exists():
            raise FileNotFoundError(f"FlyWire connectome cache not found at: {graph_path}. Run build_canonical_connectome.py first.")
        self.data: Dict[str, Any] = torch.load(graph_path, map_location="cpu", weights_only=False)
        self.cell_names: List[str] = self.data["cell_names"]
        self.name2idx: Dict[str, int] = self.data["name2idx"]
        self.metadata: Dict[str, Dict[str, str]] = self.data["metadata"]
        self.W_raw: torch.Tensor = self.data["W_raw"]
        self.W_norm: torch.Tensor = self.data["W_normalized"]
        self.spatial_kernel_bank: torch.Tensor = self.data["spatial_kernel_bank"]

    @property
    def num_neurons(self) -> int:
        return len(self.cell_names)

    def get_layer_indices(self) -> Dict[str, List[int]]:
        """Groups neuron indices by biological layer."""
        layers = {"Retina": [], "Lamina": [], "Medulla": [], "Lobula_Plate": []}
        for name, idx in self.name2idx.items():
            layer = self.metadata[name]["layer"]
            if layer in layers:
                layers[layer].append(idx)
        return layers

    def get_motion_output_indices(self) -> Dict[str, int]:
        """Returns indices for T4 (ON) and T5 (OFF) direction selective outputs."""
        motion_cells = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
        return {name: self.name2idx[name] for name in motion_cells if name in self.name2idx}

    def get_spatial_kernel(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Returns the [N_post, N_pre, 3, 3] kernel bank for 2D spatial convolution.
        Frozen (requires_grad = False) to represent zero-parameter biological inductive bias.
        """
        return self.spatial_kernel_bank.clone().detach().to(device)

def get_flywire_loader(path: Path = DEFAULT_GRAPH_PATH) -> FlyWireConnectomeLoader:
    return FlyWireConnectomeLoader(path)

if __name__ == "__main__":
    loader = get_flywire_loader()
    print("FlyWireConnectomeLoader initialized successfully.")
    print(f"Neurons loaded: {loader.num_neurons}")
    layers = loader.get_layer_indices()
    for layer, idxs in layers.items():
        names = [loader.cell_names[i] for i in idxs]
        print(f"  - {layer:<12}: {len(idxs)} cells -> {names}")
    motion = loader.get_motion_output_indices()
    print(f"Motion outputs : {motion}")
    k = loader.get_spatial_kernel()
    print(f"Spatial Kernel : shape={k.shape}, dtype={k.dtype}, non-zeros={(k != 0).sum().item()}")
