"""Whisper-specific metadata on the native EAGLE-3 transformer."""

from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig


class WhisperEagle3DraftModel(Eagle3DraftModel):
    """Native EAGLE-3 with a linear-chain verification budget.

    The builder replaces the frozen verifier RMSNorm with Whisper's LayerNorm.
    Keeping that adaptation outside the native model preserves text-model behavior.
    """

    def __init__(self, config: Eagle3SpeculatorConfig, *, block_size: int = 4):
        if block_size < 2:  # noqa: PLR2004 -- anchor plus at least one proposal
            raise ValueError("EAGLE-3 needs block_size >= 2")
        super().__init__(config)
        self.block_size = block_size
