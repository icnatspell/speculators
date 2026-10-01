"""In-memory teacher features for experimental Whisper draft training."""

import torch
from transformers import WhisperForConditionalGeneration

_TOKEN_DIMS = 2
_AUDIO_DIMS = 3


class WhisperFeatureAdapter:
    """Extract one unpadded decoder sequence without writing hidden states.

    Layer IDs index HF decoder hidden states: zero is the embedding output;
    the last index is the normalized final decoder output. The separate
    ``verifier_last_hidden_states`` field is *pre*-final-norm, so applying the
    teacher's LayerNorm and output projection reconstructs teacher logits.
    """

    def __init__(
        self,
        teacher: WhisperForConditionalGeneration,
        target_layer_ids: list[int],
    ):
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
    def extract(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        prompt_length: int,
        loss_mask: torch.Tensor | None = None,
        encoder_outputs=None,
    ) -> dict[str, torch.Tensor]:
        """Return batched DFlash fields, with loss masks indexed by target token.

        ``decoder_input_ids`` includes the prompt and generated continuation.
        Padding/batching multiple utterances is deliberately unsupported here.
        """
        if decoder_input_ids.ndim != _TOKEN_DIMS or decoder_input_ids.shape[0] != 1:
            raise ValueError("Expected one unpadded decoder sequence [1, length]")
        if input_features.ndim != _AUDIO_DIMS or input_features.shape[0] != 1:
            raise ValueError("Expected one audio feature tensor [1, mel, frames]")
        length = decoder_input_ids.shape[1]
        if not 1 <= prompt_length < length:
            raise ValueError("Prompt must be nonempty and followed by generated tokens")
        final_inputs = []

        def capture_final_input(_module, args):
            final_inputs.append(args[0].detach())

        handle = self.teacher.model.decoder.layer_norm.register_forward_pre_hook(
            capture_final_input
        )
        try:
            output = self.teacher(
                input_features=input_features if encoder_outputs is None else None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(final_inputs) != 1:
            raise RuntimeError("Expected exactly one final decoder normalization")
        if loss_mask is None:
            loss_mask = torch.zeros_like(decoder_input_ids, dtype=torch.float32)
            # DFlash rolls teacher logits by one: target token t uses logits t-1.
            loss_mask[:, prompt_length:] = 1
        else:
            if loss_mask.shape != decoder_input_ids.shape:
                raise ValueError("loss_mask must match decoder_input_ids")
            loss_mask = loss_mask.to(
                device=decoder_input_ids.device, dtype=torch.float32
            )
        return {
            "input_ids": decoder_input_ids.detach(),
            "hidden_states": torch.cat(
                [output.decoder_hidden_states[i] for i in self.target_layer_ids],
                dim=-1,
            ).detach(),
            "verifier_last_hidden_states": final_inputs[0],
            "loss_mask": loss_mask,
            "document_ids": torch.zeros_like(decoder_input_ids),
            "position_ids": torch.arange(length, device=decoder_input_ids.device)[None],
        }
