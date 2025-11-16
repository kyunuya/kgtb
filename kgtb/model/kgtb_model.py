from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder


class ResidualQuantizer(nn.Module):
    def __init__(self, num_layers: int, codebook_sizes: list[int], dim: int, beta: float = 0.25):
        super().__init__()
        self.num_layers = num_layers
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(size, dim) for size in codebook_sizes])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        all_indices = []
        all_quantized_vectors = []
        total_loss = 0.0

        for layer in range(self.num_layers):
            codebook = self.codebooks[layer].weight
            # This is a key stabilization technique for quantization models.
            codebook = F.normalize(codebook, p=2.0, dim=-1, eps=1e-12)
            distances = torch.cdist(residual, codebook, p=2.0)
            indices = torch.argmin(distances, dim=-1)
            all_indices.append(indices)

            quantized_vectors = self.codebooks[layer](indices)
            all_quantized_vectors.append(quantized_vectors)

            # --- L_RQ Calculation ---
            # This implements the formula from the "Training" section:
            # L_RQ = sum(||sg[z_l] - b||^2 + beta * ||sg[b] - z_l||^2)
            codebook_loss = F.mse_loss(residual, quantized_vectors.detach())
            commitment_loss = F.mse_loss(residual.detach(), quantized_vectors)
            total_loss += codebook_loss + self.beta * commitment_loss

            # --- Straight-Through Estimator ---
            # Gradients are copied from quantized_vectors to the residual.
            ste_vectors = residual + (quantized_vectors - residual).detach()

            # Update the residual for the next layer.
            residual = residual - ste_vectors

        stru_ids = torch.stack(all_indices, dim=1)

        # --- Decode `hat(h)` ---
        # As per the "Knowledge Graph Reconstruction" section:
        # hat(h)_i = sum_{l=1 to L} b^e_{l, n^e_l}
        decoded_vectors = torch.stack(all_quantized_vectors, dim=0).sum(dim=0)

        return stru_ids, decoded_vectors, total_loss


class KGTBModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        entity_type_map: Dict[str, torch.Tensor],
        dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.entity_type_map = entity_type_map
        self.num_layers = num_layers
        self.dim = dim

        self.encoder = GraphEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=dim,
            num_layers=num_layers,
        )

        MAX_CODEBOOK_SIZE = 4096

        codebook_sizes = {
            "poi": [min(MAX_CODEBOOK_SIZE, 8 ** (4 * (layer - 1))) for layer in range(1, num_layers + 1)],
            "user": [min(MAX_CODEBOOK_SIZE, 8 ** (4 * (layer - 1))) for layer in range(1, num_layers + 1)],
            "region": [min(MAX_CODEBOOK_SIZE, 8 ** (2 * (layer - 1))) for layer in range(1, num_layers + 1)],
            "category": [min(MAX_CODEBOOK_SIZE, 8 ** (2 * (layer - 1))) for layer in range(1, num_layers + 1)],
        }

        self.quantizers = nn.ModuleDict({
            entity_type: ResidualQuantizer(num_layers, sizes, dim)
            for entity_type, sizes in codebook_sizes.items()
        })

        self.relation_weights = nn.Parameter(torch.randn(num_relations, dim, dim))
        self.norm = nn.LayerNorm(dim)

    def _score_triples(self, h_head: torch.Tensor, h_tail: torch.Tensor, rel_indices: torch.Tensor) -> torch.Tensor:
        W_r = self.relation_weights[rel_indices]
        h_head = h_head.unsqueeze(1)
        head_rel_prod = torch.bmm(h_head, W_r)

        h_tail = h_tail.unsqueeze(2)

        score = torch.bmm(head_rel_prod, h_tail).squeeze()

        return score

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, positive_triples: torch.Tensor) -> torch.Tensor:
        h = self.encoder(edge_index, edge_type)
        h = self.norm(h)

        if torch.isnan(h).any():
            print("NaN detected in encoder output!")

        decoded_vectors = torch.zeros_like(h)
        total_l_rq = 0.0
        for entity_type, node_indices in self.entity_type_map.items():
            if node_indices.numel() > 0:
                h_type = h[node_indices]
                _, decoded_h, l_rq = self.quantizers[entity_type](h_type)

                if torch.isnan(decoded_h).any():
                    print(f"NaN detected in decoded vectors for {entity_type}!")
                if torch.isnan(l_rq).any():
                    print(f"NaN detected in L_RQ for {entity_type}!")

                decoded_vectors[node_indices] = decoded_h
                total_l_rq += l_rq

        num_pos = positive_triples.shape[0]
        corrupted_tails = torch.randint(0, self.num_nodes, (num_pos,), device=h.device)
        negative_triples = positive_triples.clone()
        negative_triples[:, 2] = corrupted_tails

        all_triples = torch.cat([positive_triples, negative_triples], dim=0)
        labels = torch.cat([torch.ones(num_pos), torch.zeros(num_pos)], dim=0).to(h.device)

        h_head = decoded_vectors[all_triples[:, 0]]
        h_tail = decoded_vectors[all_triples[:, 2]]
        relations = all_triples[:, 1]

        h_head = self.norm(h_head)
        h_tail = self.norm(h_tail)

        h_head = torch.tanh(h_head)
        h_tail = torch.tanh(h_tail)

        scores = self._score_triples(h_head, h_tail, relations)

        if torch.isnan(scores).any():
            print("NaN detected in scores!")

        l_kg = F.binary_cross_entropy_with_logits(scores, labels)

        if torch.isnan(l_kg).any():
            print("NaN detected in L_KG!")

        total_loss = l_kg + total_l_rq
        return total_loss

    @torch.no_grad()
    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        self.eval()
        h = self.encoder(edge_index, edge_type)

        all_stru_ids = torch.zeros(self.num_nodes, self.num_layers, dtype=torch.long, device=h.device)

        for entity_type, node_indices in self.entity_type_map.items():
            if node_indices.numel() > 0:
                h_type = h[node_indices]
                stru_ids, _, _ = self.quantizers[entity_type](h_type)
                all_stru_ids[node_indices] = stru_ids

        return all_stru_ids
