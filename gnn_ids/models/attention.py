"""Post-GCN attention over three per-flow components (notebook section 6)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PostGCNAttention(nn.Module):
    """Lightweight attention over three per-flow components, applied AFTER the GCN.

    Components (each in R^latent_dim):
        s1 = h_src   : source-node embedding from the GCN
        s2 = h_dst   : destination-node embedding from the GCN
        s3 = z_edge  : the flow's 49 features projected to latent_dim

    ONE linear layer scores the concatenated triple, so each alpha is
    input-dependent and aware of the other two components. Softmax over the 3
    components gives alpha1 + alpha2 + alpha3 = 1 per flow.

    Output: z_flow = alpha1*s1 + alpha2*s2 + alpha3*s3  in R^latent_dim.

    Not GAT: no neighbourhood attention, no edge_index, no message passing.
    Never receives labels.
    """

    N_COMPONENTS = 3

    def __init__(self, node_dim=64, edge_in_dim=49, latent_dim=64,
                 use_layernorm=False, dropout=0.0):
        super().__init__()
        assert node_dim == latent_dim, "weighted sum needs matching dimensions"
        self.latent_dim = latent_dim
        self.use_layernorm = use_layernorm

        self.edge_proj = nn.Linear(edge_in_dim, latent_dim)      # 49 -> 64

        if use_layernorm:
            self.norm_src = nn.LayerNorm(latent_dim)
            self.norm_dst = nn.LayerNorm(latent_dim)
            self.norm_edge = nn.LayerNorm(latent_dim)

        # The lightweight attention: one linear layer -> 3 scores -> softmax
        self.score = nn.Linear(self.N_COMPONENTS * latent_dim, self.N_COMPONENTS)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)   # start near-uniform alphas (~1/3 each)
        if self.use_layernorm:
            for m in (self.norm_src, self.norm_dst, self.norm_edge):
                m.reset_parameters()

    def forward(self, h_src, h_dst, x_edge, return_components=False):
        """h_src, h_dst: [B, latent_dim]   x_edge: [B, edge_in_dim]"""
        z_edge = F.relu(self.edge_proj(x_edge))                 # [B, D]

        s_src, s_dst, s_edge = h_src, h_dst, z_edge
        if self.use_layernorm:
            s_src = self.norm_src(s_src)
            s_dst = self.norm_dst(s_dst)
            s_edge = self.norm_edge(s_edge)

        context = torch.cat([s_src, s_dst, s_edge], dim=-1)     # [B, 3D]
        alpha = F.softmax(self.score(context), dim=-1)          # [B, 3]

        components = torch.stack([s_src, s_dst, s_edge], dim=1)          # [B, 3, D]
        z_flow = (alpha.unsqueeze(-1) * components).sum(dim=1)           # [B, D]
        z_flow = self.dropout(z_flow)

        if return_components:
            return z_flow, alpha, components
        return z_flow, alpha


def gather_flow_inputs(graph, node_emb, flow_idx, device):
    """For original flows `flow_idx`, gather (h_src, h_dst, x_edge).

    graph    : training ``Data`` object (flow_src / flow_dst / flow_attr stay on CPU)
    node_emb : [num_nodes, D] on `device`, output of the GCN
    flow_idx : LongTensor of positions into the N original flows (CPU)

    Reverse message-passing copies are unreachable: flow_src/flow_dst/flow_attr
    each have exactly N rows. Labels are NOT gathered here.
    """
    assert int(flow_idx.max()) < graph.flow_src.shape[0], "Index beyond the original flow count"
    assert int(flow_idx.min()) >= 0

    src = graph.flow_src[flow_idx].to(device, non_blocking=True)
    dst = graph.flow_dst[flow_idx].to(device, non_blocking=True)
    x_edge = graph.flow_attr[flow_idx].to(device, non_blocking=True)

    return node_emb[src], node_emb[dst], x_edge
