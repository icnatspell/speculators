"""Cached linear-chain EAGLE-3 proposals for the shared Whisper verifier."""

import torch
from transformers import DynamicCache

from speculators.models.eagle3.whisper import WhisperEagle3DraftModel


class Eagle3WhisperProposal:
    """Keep verified feature/token KV pairs; discard every speculative suffix.

    In pair t, the shifted token x[t+1] accompanies teacher features h[t].
    Generated steps feed their predicted latent state back without FC projection.
    Before the next round, cached hypothetical features are discarded so newly
    accepted tokens receive the actual teacher features. No teacher KV is owned
    by this proposer; the shared verifier handles its own rejection rollback.
    """

    def __init__(self, draft: WhisperEagle3DraftModel, *, cache_context=True):
        self.draft = draft
        self.cache_context = cache_context
        self.cache = DynamicCache()

    def _project_features(self, features):
        draft = self.draft
        if draft.input_norm is not None:
            features = draft.input_norm(features)
        if draft.fc_norm is not None:
            features = torch.cat(
                [
                    norm(chunk)
                    for norm, chunk in zip(
                        draft.fc_norm,
                        features.chunk(len(draft.fc_norm), dim=-1),
                        strict=True,
                    )
                ],
                dim=-1,
            )
        return draft.fc(features)

    def _forward(self, hidden, tokens):
        draft = self.draft
        embeddings = draft.embed_tokens(tokens)
        hidden = torch.cat([embeddings, hidden], dim=-1)
        start = self.cache.get_seq_length()
        positions = torch.arange(
            start + 1, start + tokens.shape[1] + 1, device=tokens.device
        )[None]
        cache_positions = positions[0] - 1
        rotary = draft.rotary_emb(hidden, positions)
        # An explicit mask is required for multi-token suffix prefill with past KV.
        keys = torch.arange(start + tokens.shape[1], device=tokens.device)
        allowed = keys[None] <= cache_positions[:, None]
        mask = torch.zeros(allowed.shape, device=tokens.device, dtype=hidden.dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]
        for layer in draft.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=self.cache,
                cache_position=cache_positions,
                position_embeddings=rotary,
                use_cache=True,
            )
        hidden = hidden[:, -1:]
        latent = draft.norm(hidden) if draft.config.norm_output else hidden
        logits = draft.lm_head(
            latent if draft.config.norm_output else draft.norm(hidden)
        )
        return latent, logits[:, -1]

    @torch.no_grad()
    def prefill(self, context, prefix):
        """Prepare verified prompt KV before generation timing begins."""
        if not self.cache_context or prefix.shape[1] < 2:  # noqa: PLR2004
            return
        start = self.cache.get_seq_length()
        keep = prefix.shape[1] - 1
        if start < keep:
            device = self.draft.embed_tokens.weight.device
            self._forward(
                self._project_features(context[:, start:keep].to(device)),
                prefix[:, start + 1 :].to(device),
            )

    @torch.no_grad()
    def __call__(self, context, prefix, anchor, *, budget, eos, select_token):
        if context.shape[1] != prefix.shape[1]:
            raise ValueError("EAGLE-3 teacher context must match the verified prefix")
        if budget < 2:  # noqa: PLR2004 -- anchor plus proposal
            return anchor
        if not self.cache_context:
            self.cache = DynamicCache()
        device = self.draft.embed_tokens.weight.device
        # Prefix shortening is supported; the proposer cannot retain future KV.
        keep = prefix.shape[1] - 1
        if self.cache.get_seq_length() > keep:
            self.cache.crop(keep - self.cache.get_seq_length())
        start = self.cache.get_seq_length()
        tokens = torch.cat([prefix[:, 1:], anchor], dim=1).to(device)
        hidden = self._project_features(context[:, start:].to(device))
        candidates = [anchor]
        try:
            latent, logits = self._forward(hidden, tokens[:, start:])
            for position in range(min(self.draft.block_size - 1, budget - 1)):
                candidate_prefix = torch.cat([prefix, *candidates], dim=1)
                token = select_token(candidate_prefix, logits.to(prefix.device))
                candidates.append(token)
                if token.item() == eos or position + 1 == min(
                    self.draft.block_size - 1, budget - 1
                ):
                    break
                latent, logits = self._forward(latent, token.to(device))
        finally:
            # KV after `keep` was computed using draft latents, not verified features.
            removed = self.cache.get_seq_length() - keep
            if removed:
                self.cache.crop(-removed)
        return torch.cat(candidates, dim=1)
