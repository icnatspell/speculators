"""Utility functions for DFlash draft model."""

import torch


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
    *,
    document_ids: torch.Tensor | None = None,
    allow_partial_blocks: bool = False,
    anchors_per_document: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [1, total_seq_len]
        n: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded)

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [num_anchors]
            - anchor_valid: Boolean mask for valid anchors [num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    if allow_partial_blocks:
        valid_mask[:, -1] = False
        valid_mask[:, :-1] &= loss_mask[:, 1:].bool()
        if document_ids is not None:
            valid_mask[:, :-1] &= document_ids[:, :-1] == document_ids[:, 1:]
            valid_mask &= document_ids != -1
    else:
        valid_mask[:, -block_size:] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())

    # Constrain value of k for torch dynamo
    torch._check(k <= valid_indices.numel())  # noqa: SLF001
    torch._check(k >= 0)  # noqa: SLF001

    if anchors_per_document and document_ids is not None:
        groups = []
        for doc in torch.unique(document_ids[0, valid_indices]).tolist():
            group = valid_indices[document_ids[0, valid_indices] == doc]
            groups.append(group[torch.randperm(group.numel(), device=device)])
        interleaved = [
            group[i : i + 1]
            for i in range(num_anchors)
            for group in groups
            if i < group.numel()
        ]
        chosen = torch.cat(interleaved)[:k] if interleaved else valid_indices
        anchors[:k] = torch.sort(chosen).values
        anchor_valid[:k] = True
        return anchors, anchor_valid

    perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
    # Contiguous anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    anchors[:k] = torch.sort(torch.gather(valid_indices, 0, perm[:k])).values
    anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]
