from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_REFERENCE_BYTES = 32 * 1024 * 1024


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=12000)
    voice_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    exaggeration: float = Field(default=0.5, ge=0, le=1.5)
    cfg_weight: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=0.8, ge=0.1, le=2)
    repetition_penalty: float = Field(default=1.2, ge=1, le=3)
    min_p: float = Field(default=0.05, ge=0, le=1)
    top_p: float = Field(default=1.0, gt=0, le=1)
    autocast: Literal["off", "fp16", "bf16"] = "off"
    seed: int | None = Field(default=None, ge=-1, le=2**32 - 1)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def plain_text(cls, value):
        if not value.strip():
            raise ValueError("Text must contain speech")
        if any(token in value for token in ("[START]", "[STOP]", "[SPACE]")):
            raise ValueError(
                "Input must be speech text without tokenizer control tokens"
            )
        return value.strip()
