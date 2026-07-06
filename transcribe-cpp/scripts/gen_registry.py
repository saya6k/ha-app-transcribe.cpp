#!/usr/bin/env python3
"""Generate wyoming_transcribe_cpp/registry.json from upstream hf_cards.

Usage:
    python3 scripts/gen_registry.py /path/to/transcribe.cpp/checkout

Reads every ``scripts/hf_cards/*.yaml`` in the pinned upstream checkout,
drops non-commercial-licensed models (SPEC boundary), writes the registry
JSON, and rewrites the ``model: list(...)`` schema line in config.yaml so
the dropdown always matches the registry (enforced by a unit test).

Requires PyYAML (dev-time only; the app itself never parses the cards).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

APP_DIR = Path(__file__).parent.parent
REGISTRY_JSON = APP_DIR / "wyoming_transcribe_cpp" / "registry.json"
CONFIG_YAML = APP_DIR / "config.yaml"

EXCLUDED_LICENSES = {"cc-by-nc-4.0"}  # non-commercial — SPEC "never"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cards_dir = Path(sys.argv[1]) / "scripts" / "hf_cards"
    if not cards_dir.is_dir():
        raise SystemExit(f"No hf_cards directory at {cards_dir}")

    registry: dict[str, dict] = {}
    excluded: list[str] = []
    for card_path in sorted(cards_dir.glob("*.yaml")):
        card = yaml.safe_load(card_path.read_text())
        if not isinstance(card, dict) or "target_repo" not in card:
            continue
        name = card_path.stem
        if card["license"] in EXCLUDED_LICENSES:
            excluded.append(f"{name} ({card['license']})")
            continue
        registry[name] = {
            "repo": card["target_repo"],
            "license": card["license"],
            "streaming": bool((card.get("capabilities") or {}).get("streaming")),
            "quants": {q["name"]: q["filename"] for q in card["quants"]},
            "languages": card.get("languages") or [],
        }

    REGISTRY_JSON.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    print(f"registry.json: {len(registry)} models (excluded: {excluded})")

    model_list = "|".join(sorted(registry))
    config = CONFIG_YAML.read_text()
    config, n = re.subn(
        r"^  model: list\([^)]*\)$", f"  model: list({model_list})", config, flags=re.M
    )
    if n != 1:
        raise SystemExit("config.yaml: expected exactly one 'model: list(...)' line")
    CONFIG_YAML.write_text(config)
    print("config.yaml model list updated")


if __name__ == "__main__":
    main()
