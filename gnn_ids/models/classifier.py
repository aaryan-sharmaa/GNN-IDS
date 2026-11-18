"""Flow classifier head and the three-module bundle of the main GNN (notebook sections 5-7)."""
from dataclasses import dataclass

import torch.nn as nn

from ..config import (ATT_DROPOUT, ATT_LATENT_DIM, ATT_SEED, ATT_USE_LAYERNORM, CLS_DROPOUT,
                      CLS_HIDDEN_DIM, GCN_HIDDEN_DIM, GCN_OUT_DIM, MODEL_SEED, N_FEATURES,
                      N_PARAMS_ATTENTION, N_PARAMS_TOTAL, RANDOM_SEED)
from ..utils import seed_torch
from .attention import PostGCNAttention
from .gcn import GCNEncoder


class FlowClassifier(nn.Module):
    """Maps the 64-D attention output of one flow to a single logit."""

    def __init__(self, in_dim=64, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z_flow):
        h = self.dropout(self.relu(self.fc1(z_flow)))
        return self.fc2(h).squeeze(-1)      # [B]


@dataclass
class GnnModules:
    """The three trainable components of the GNN + Post-GCN attention model."""
    gcn: GCNEncoder
    attention: PostGCNAttention
    classifier: FlowClassifier

    def train(self):
        self.gcn.train(); self.attention.train(); self.classifier.train()

    def eval(self):
        self.gcn.eval(); self.attention.eval(); self.classifier.eval()

    def parameters(self):
        return (list(self.gcn.parameters()) + list(self.attention.parameters())
                + list(self.classifier.parameters()))


def build_gnn_modules(node_in_dim, device, staged_seeding=True) -> GnnModules:
    """Instantiate GCN, attention and classifier.

    staged_seeding=True  (UNSW, notebook sections 5, 6, 7.1): the seed is reset before each
                         module is created.
    staged_seeding=False (CSE, notebook section 10.6-R): one seed, then the three modules
                         are created in sequence.
    """
    seed_torch(MODEL_SEED if staged_seeding else RANDOM_SEED)
    gcn = GCNEncoder(in_channels=node_in_dim, hidden_channels=GCN_HIDDEN_DIM,
                     out_channels=GCN_OUT_DIM).to(device)

    if staged_seeding:
        seed_torch(ATT_SEED)
    attention = PostGCNAttention(node_dim=GCN_OUT_DIM, edge_in_dim=N_FEATURES,
                                 latent_dim=ATT_LATENT_DIM, use_layernorm=ATT_USE_LAYERNORM,
                                 dropout=ATT_DROPOUT).to(device)
    assert sum(p.numel() for p in attention.parameters()) == N_PARAMS_ATTENTION

    if staged_seeding:
        seed_torch(RANDOM_SEED)
    classifier = FlowClassifier(in_dim=ATT_LATENT_DIM, hidden_dim=CLS_HIDDEN_DIM,
                                dropout=CLS_DROPOUT).to(device)

    modules = GnnModules(gcn, attention, classifier)
    n_params = sum(p.numel() for p in modules.parameters())
    assert n_params == N_PARAMS_TOTAL, f"Parameter count changed: {n_params}"
    return modules
