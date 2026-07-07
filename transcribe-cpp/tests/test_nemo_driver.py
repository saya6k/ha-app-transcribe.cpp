"""Unit tests for the parakeet direct-load driver (fake script, no torch)."""

import sys
from pathlib import Path

FAKE_SCRIPT = '''
VARIANT_PROFILES = {
    "a": {"head_kind": "rnnt"},
    "b": {"head_kind": "tdt", "prefer_direct_load": False},
}

def _load_nemo_archive_directly(path):
    return "upstream-loader"

CALLS = {}

def main(argv):
    CALLS["argv"] = argv
    CALLS["profiles"] = {k: v.get("prefer_direct_load") for k, v in VARIANT_PROFILES.items()}
    CALLS["loader"] = _load_nemo_archive_directly
    return 0

if __name__ == "__main__":
    raise SystemExit("must never run on import")
'''


def run_driver(tmp_path, monkeypatch):
    script = tmp_path / "convert-parakeet.py"
    script.write_text(FAKE_SCRIPT)
    driver_dir = Path(__file__).parents[1] / "wyoming_transcribe_cpp"
    monkeypatch.setattr(
        sys, "argv", ["nemo_driver.py", str(script), "model.nemo", "out.gguf"]
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nemo_driver_under_test", driver_dir / "nemo_driver.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    try:
        driver.main()
    except SystemExit as e:
        assert e.code == 0
    return driver


class TestNemoDriver:
    def test_flags_flipped_loader_replaced_main_called(self, tmp_path, monkeypatch):
        driver = run_driver(tmp_path, monkeypatch)
        mod = sys.modules.get("convert_parakeet")
        assert mod is not None
        calls = mod.CALLS
        assert calls["argv"][1:] == ["model.nemo", "out.gguf"]
        assert calls["profiles"] == {"a": True, "b": True}
        # the byte-copying upstream loader is swapped for the streaming one
        assert calls["loader"].__name__ == "_streaming_load_nemo_archive"

    def test_streaming_loader_reads_a_real_archive(self, tmp_path, monkeypatch):
        import io
        import pickle
        import tarfile

        driver = run_driver(tmp_path, monkeypatch)
        # a fake torch module so the loader is testable without torch
        import types

        fake_torch = types.ModuleType("torch")

        def fake_load(f, **kw):
            if isinstance(f, str):
                with open(f, "rb") as fh:
                    return pickle.load(fh)
            return pickle.load(f)

        fake_torch.load = fake_load
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        nemo = tmp_path / "model.nemo"
        with tarfile.open(nemo, "w") as tf:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            add("./model_config.yaml", b"foo: 1\n")
            add("./model_weights.ckpt", pickle.dumps({"w": [1, 2, 3]}))
            add("./abc_tokenizer.model", b"SPMPROTO")

        cfg, sd, sp = driver._streaming_load_nemo_archive(nemo)
        assert cfg == {"foo": 1}
        assert sd == {"w": [1, 2, 3]}
        assert sp == b"SPMPROTO"
