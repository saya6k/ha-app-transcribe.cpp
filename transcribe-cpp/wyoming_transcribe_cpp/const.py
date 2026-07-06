"""Shared constants."""

PORT = 10380
MODELS_DIR = "/data/models"

# GGUF precisions ordered smallest -> largest. Quant resolution walks this
# list upward from the requested precision (never downward — a smaller
# precision than asked for would silently degrade quality).
QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32"]

# config.yaml option values (lowercase) -> GGUF precision names.
QUANT_ALIASES = {q.lower(): q for q in QUANT_ORDER}
