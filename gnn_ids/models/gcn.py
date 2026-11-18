"""2-layer GCN node encoder (notebook section 5)."""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """2-layer GCN producing node embeddings.

    Consumes ONLY node features and the message-passing edge_index.
    Flow edge features and flow labels never enter this module.
    No edge_weight: flow multiplicity in edge_index acts as the weighting,
    handled by GCNConv's internal symmetric normalization.
    """

    def __init__(self, in_channels=1, hidden_channels=64, out_channels=64):
        super().__init__()
        # cached=False: gcn_norm is recomputed each forward pass. Our graph is
        # fixed, so caching would be valid, but recompute is cheap and avoids a
        # stale cache if the graph is ever rebuilt (e.g. a different fraction).
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h  # [num_nodes, out_channels]
