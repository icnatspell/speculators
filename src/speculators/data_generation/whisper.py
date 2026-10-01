"""Ephemeral, batched Whisper decoder features without vocabulary projection."""

import threading

import torch
from transformers import WhisperForConditionalGeneration


class WhisperFeatureAdapter:
    """Capture selected HF hidden-state indices and pre-final-norm states.

    Index zero is the embedding/position output; index N is the normalized
    output of an N-layer decoder. Other indices are inputs to decoder layer i.
    Only requested states are retained. Callers own exclusive teacher access.
    """

    def __init__(self, teacher: WhisperForConditionalGeneration, target_layer_ids):
        num_layers = teacher.config.decoder_layers
        if (
            not target_layer_ids
            or len(set(target_layer_ids)) != len(target_layer_ids)
            or min(target_layer_ids) < 0
            or max(target_layer_ids) > num_layers
        ):
            raise ValueError(f"Layer IDs must be distinct and within [0, {num_layers}]")
        self.teacher = teacher.eval().requires_grad_(False)
        self.target_layer_ids = list(target_layer_ids)

    @torch.no_grad()
    def extract_batch(
        self,
        input_features,
        sequences,
        *,
        prompt_lengths,
        loss_masks=None,
        encoder_outputs=None,
        pad_to_multiple=1,
    ):
        """Run one padded teacher batch, then return independent unpadded rows."""
        if input_features.ndim != 3 or input_features.shape[0] != len(sequences):  # noqa: PLR2004 -- [B, mel, frames]
            raise ValueError("Expected matching audio and sequence batch dimensions")
        if len(prompt_lengths) != len(sequences) or not sequences:
            raise ValueError("Expected one prompt length per sequence")
        lengths = [sequence.numel() for sequence in sequences]
        if any(not 1 <= p < n for p, n in zip(prompt_lengths, lengths, strict=True)):
            raise ValueError("Prompt must be nonempty and followed by generated tokens")
        width = (
            (max(lengths) + pad_to_multiple - 1) // pad_to_multiple
        ) * pad_to_multiple
        width = min(width, self.teacher.config.max_target_positions)
        if width < max(lengths):
            raise ValueError("Decoder sequence exceeds the teacher position limit")
        device = self.teacher.device
        tokens = torch.full(
            (len(sequences), width),
            self.teacher.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(tokens)
        for index, sequence in enumerate(sequences):
            tokens[index, : lengths[index]] = sequence.flatten().to(device)
            attention[index, : lengths[index]] = 1
        selected, final_inputs = {}, []
        owner = threading.get_ident()

        def capture(index):
            def hook(_module, args, kwargs):
                if threading.get_ident() == owner:
                    selected[index] = (
                        args[0] if args else kwargs["hidden_states"]
                    ).detach()

            return hook

        def capture_final(_module, args):
            if threading.get_ident() == owner:
                final_inputs.append(args[0].detach())

        decoder = self.teacher.model.decoder
        handles = [decoder.layer_norm.register_forward_pre_hook(capture_final)]
        for index in self.target_layer_ids:
            if index < len(decoder.layers):
                handles.append(
                    decoder.layers[index].register_forward_pre_hook(
                        capture(index),
                        with_kwargs=True,
                    )
                )
        try:
            # WhisperModel omits the expensive, unused full-vocabulary logits.
            output = self.teacher.model(
                input_features=input_features.to(
                    device=device, dtype=self.teacher.dtype
                )
                if encoder_outputs is None
                else None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=tokens,
                decoder_attention_mask=attention,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(final_inputs) != 1:
            raise RuntimeError("Expected exactly one final decoder normalization")
        selected[len(decoder.layers)] = output.last_hidden_state
        hidden = torch.cat([selected[index] for index in self.target_layer_ids], dim=-1)
        rows = []
        for index, length in enumerate(lengths):
            ids = tokens[index : index + 1, :length]
            if loss_masks is None:
                mask = torch.zeros_like(ids, dtype=torch.float32)
                mask[:, prompt_lengths[index] :] = 1
            else:
                mask = (
                    loss_masks[index]
                    .reshape(1, -1)
                    .to(device=device, dtype=torch.float32)
                )
                if mask.shape != ids.shape:
                    raise ValueError("loss_mask must match decoder_input_ids")
            # Clone slices so an individual row does not retain batch padding.
            rows.append(
                {
                    "input_ids": ids.clone(),
                    "hidden_states": hidden[index : index + 1, :length]
                    .detach()
                    .clone(),
                    "verifier_last_hidden_states": final_inputs[0][
                        index : index + 1, :length
                    ].clone(),
                    "loss_mask": mask,
                    "document_ids": torch.zeros_like(ids),
                    "position_ids": torch.arange(length, device=device)[None],
                }
            )
        return rows

    def extract(
        self,
        input_features,
        decoder_input_ids,
        *,
        prompt_length,
        loss_mask=None,
        encoder_outputs=None,
    ):
        """Compatibility entry point for one unpadded utterance."""
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != 1:  # noqa: PLR2004 -- [B, tokens]
            raise ValueError("Expected one unpadded decoder sequence [1, length]")
        return self.extract_batch(
            input_features,
            [decoder_input_ids[0]],
            prompt_lengths=[prompt_length],
            loss_masks=None if loss_mask is None else [loss_mask],
            encoder_outputs=encoder_outputs,
        )[0]
