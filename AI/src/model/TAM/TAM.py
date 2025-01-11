import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalAggregationModule"]

class DisentangledAttention(nn.Module):
    def __init__(self, embed_dim, max_relative_position):
        super(DisentangledAttention, self).__init__()
        self.embed_dim = embed_dim
        self.max_relative_position = max_relative_position

        # Projection matrices for content-to-content attention
        self.Wq_ct = nn.Linear(embed_dim, embed_dim)
        self.Wk_ct = nn.Linear(embed_dim, embed_dim)
        self.Wv_ct = nn.Linear(embed_dim, embed_dim)

        # Projection matrices for relative position attention
        self.Wq_r = nn.Linear(embed_dim, embed_dim)
        self.Wk_r = nn.Linear(embed_dim, embed_dim)

        # Shared relative position embeddings
        self.relative_position_embeddings = nn.Embedding(2 * max_relative_position, embed_dim)

    def forward(self, Z):
        batch_size, seq_len, embed_dim = Z.size()
        assert embed_dim == self.embed_dim, "Embedding dimensions do not match"

        # Compute content queries, keys, and values
        Q_ct = self.Wq_ct(Z)  # (batch_size, seq_len, embed_dim)
        K_ct = self.Wk_ct(Z)  # (batch_size, seq_len, embed_dim)
        V_ct = self.Wv_ct(Z)  # (batch_size, seq_len, embed_dim)

        # Compute relative position queries and keys
        positions = torch.arange(seq_len, dtype=torch.long, device=Z.device)
        relative_positions = positions[None, :] - positions[:, None]
        relative_positions = torch.clamp(relative_positions + self.max_relative_position, 0, 2 * self.max_relative_position - 1)

        relative_embeddings = self.relative_position_embeddings(relative_positions)
        Q_r = self.Wq_r(relative_embeddings)  # (seq_len, seq_len, embed_dim)
        K_r = self.Wk_r(relative_embeddings)  # (seq_len, seq_len, embed_dim)

        # Compute disentangled attention scores
        content_scores = torch.einsum("bqd,bkd->bqk", Q_ct, K_ct)  # Content-to-content
        position_scores = torch.einsum("bqd,qkd->bqk", Q_ct, K_r)  # Content-to-position
        relative_scores = torch.einsum("qkd,bkd->bqk", Q_r, K_ct)  # Position-to-content

        # Aggregate attention scores
        scores = content_scores + position_scores + relative_scores
        attention_weights = F.softmax(scores / (self.embed_dim ** 0.5), dim=-1)

        # Compute attention output
        output = torch.einsum("bqk,bvd->bqd", attention_weights, V_ct)
        return output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.Wq_cross = nn.Linear(embed_dim, embed_dim)
        self.Wk_cross = nn.Linear(embed_dim, embed_dim)
        self.Wv_cross = nn.Linear(embed_dim, embed_dim)

    def forward(self, Z1, Z2):
        # Compute queries, keys, and values for cross attention
        Q_cross = self.Wq_cross(Z1)
        K_cross = self.Wk_cross(Z2)
        V_cross = self.Wv_cross(Z2)

        # Compute cross-attention scores
        scores = torch.einsum("bqd,bkd->bqk", Q_cross, K_cross)
        attention_weights = F.softmax(scores / (Q_cross.size(-1) ** 0.5), dim=-1)

        # Compute cross-attention output
        output = torch.einsum("bqk,bvd->bqd", attention_weights, V_cross)
        return output

class TemporalAggregationModule(nn.Module):
    def __init__(self, embed_dim, max_relative_position, num_backbones):
        super(TemporalAggregationModule, self).__init__()
        self.num_backbones = num_backbones
        self.attention_modules = nn.ModuleList([
            DisentangledAttention(embed_dim, max_relative_position) for _ in range(num_backbones)
        ])
        self.cross_attention = CrossAttention(embed_dim)

    def forward(self, inputs):
        assert len(inputs) == self.num_backbones, "Number of inputs must match the number of backbones"

        # Apply disentangled attention for each backbone
        backbone_outputs = [self.attention_modules[i](inputs[i]) for i in range(self.num_backbones)]

        # Aggregate outputs using cross attention
        fused_output = backbone_outputs[0]
        for i in range(1, self.num_backbones):
            fused_output = self.cross_attention(fused_output, backbone_outputs[i])

        # Final aggregation by averaging
        aggregated_output = sum(backbone_outputs) / self.num_backbones
        return aggregated_output

# # Example usage
# if __name__ == "__main__":
#     embed_dim = 128
#     max_relative_position = 16
#     num_backbones = 3
#     seq_len = 50
#     batch_size = 8

#     # Create input sequences for backbones
#     inputs = [torch.rand(batch_size, seq_len, embed_dim) for _ in range(num_backbones)]

#     # Initialize and run Temporal Aggregation Module
#     tam = TemporalAggregationModule(embed_dim, max_relative_position, num_backbones)
#     output = tam(inputs)

#     print("Output shape:", output.shape)
