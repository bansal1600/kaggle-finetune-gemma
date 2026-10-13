"""Tests for kaggle/common/adapter_tools.py on a tiny Gemma-4-shaped config (CPU, seconds).

Run with: python -m pytest -q tests/test_adapter_tools.py
Works without numpy (pure-Python fallback). Optional extras, skipped when absent:
  - safetensors: the reference reader accepts our files;
  - adk_submission: the scorer's discover_adapters + VllmServer.build_cmd see rank 16;
  - the organizers' vLLM 0.19.1 wheel (VLLM_WHEEL, default: the research wheelhouse): our key
    parser agrees with vLLM's own parse_fine_tuned_lora_name + gemma4_mm weights mapper.
"""

import ast
import json
import os
import struct
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "common"))

import adapter_tools as at  # noqa: E402

# Same structure as the 31B (every 6th layer is full attention with K = V), much smaller dims.
TINY = {
    "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 12,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "num_global_key_value_heads": 1, "global_head_dim": 16, "attention_k_eq_v": True,
    "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(12)],
}
GLOBAL = {5, 11}
VLLM_WHEEL = Path(os.environ.get(
    "VLLM_WHEEL",
    "/tmp/claude-0/-home-user-kaggle-finetune-gemma/16026f0c-2bda-5281-bad9-92d73f9dba6b/scratchpad/"
    "ft_research/kwh_1012/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl"))


def make(tmp_path: Path, name: str = "main_lora", **kw) -> Path:
    return at.make_dummy(tmp_path / name, kw.pop("rank", 16), config={"text_config": TINY}, **kw)


def errors(d: Path, real: bool = False) -> list[str]:
    return at.validate(d, real=real, config=TINY, strict=False)["errors"]


def edit_tensors(d: Path, fn) -> None:
    tensors = at.read_safetensors(d / "adapter_model.safetensors")
    fn(tensors)
    at.write_safetensors(d / "adapter_model.safetensors", tensors)


def edit_config(d: Path, **fields) -> None:
    cfg = json.loads((d / "adapter_config.json").read_text())
    cfg.update(fields)
    (d / "adapter_config.json").write_text(json.dumps(cfg))


# ---- the good path ----

def test_dummy_adapter_has_the_vllm_layout(tmp_path):
    d = make(tmp_path)
    report = at.validate(d, real=False, config=TINY)
    assert report["ok"] and report["max_lora_rank"] == 16 and report["scaling"] == 2.0
    # 12 layers x 7 projections, v_proj included on the 2 global layers via the duplicate
    assert report["n_modules"] == 84 and report["n_tensors"] == 168
    tensors = at.read_safetensors(d / "adapter_model.safetensors")
    assert all(k.startswith("base_model.model.model.language_model.layers.") for k in tensors)
    assert tensors[f"{at.LORA_PREFIX}.0.self_attn.q_proj.lora_A.weight"][:2] == ("BF16", [16, 64])
    assert tensors[f"{at.LORA_PREFIX}.5.self_attn.q_proj.lora_B.weight"][1] == [64, 16]  # 4 heads x 16
    assert tensors[f"{at.LORA_PREFIX}.5.self_attn.v_proj.lora_B.weight"][1] == [16, 16]  # 1 kv head x 16
    assert tensors[f"{at.LORA_PREFIX}.0.mlp.down_proj.lora_A.weight"][1] == [16, 96]
    # Both A and B are nonzero, so the adapter visibly changes the logits.
    for key in (f"{at.LORA_PREFIX}.3.mlp.up_proj.lora_A.weight", f"{at.LORA_PREFIX}.3.mlp.up_proj.lora_B.weight"):
        assert any(tensors[key][2])
    cfg = json.loads((d / "adapter_config.json").read_text())
    assert cfg["target_modules"] == at.TARGET_REGEX and cfg["lora_alpha"] == 32
    assert cfg["bias"] == "none" and cfg["modules_to_save"] is None and cfg["rank_pattern"] == {}


def test_attention_only_fallback_passes(tmp_path):
    d = make(tmp_path, attn_only=True, rank=8)
    report = at.validate(d, real=False, config=TINY)
    assert report["targets"] == ["k_proj", "o_proj", "q_proj", "v_proj"] and report["n_modules"] == 48
    assert json.loads((d / "adapter_config.json").read_text())["target_modules"] == at.ATTN_REGEX


def test_31b_geometry_matches_the_plan():
    shapes = at.module_shapes(at.text_config())
    assert len(shapes) == 420  # 410 trained + 10 duplicated v_proj modules = 840 tensors
    assert len(at.module_shapes(at.text_config(), v_on_k_eq_v=False)) == 410
    p = at.LORA_PREFIX
    assert shapes[f"{p}.0.self_attn.q_proj"] == (5376, 8192) and shapes[f"{p}.0.self_attn.k_proj"] == (5376, 4096)
    assert shapes[f"{p}.5.self_attn.q_proj"] == (5376, 16384) and shapes[f"{p}.5.self_attn.v_proj"] == (5376, 2048)
    assert shapes[f"{p}.5.self_attn.o_proj"] == (16384, 5376) and shapes[f"{p}.59.mlp.down_proj"] == (21504, 5376)
    assert at.k_eq_v_layers(at.text_config()) == set(range(5, 60, 6))


def test_bf16_rounding():
    vals = [1.0, -2.5, 1.0 + 2**-8, 1.0 + 3 * 2**-8]  # the last two are halfway cases (round to even)
    raw = struct.pack("<4f", *vals)
    expected = struct.pack("<4H", 0x3F80, 0xC020, 0x3F80, 0x3F82)
    assert at.to_bf16("F32", raw) == expected
    assert at._floats_to_bf16(vals) == expected


# ---- dup_k_as_v ----

def test_missing_v_duplicate_fails_then_dup_k_as_v_fixes_it(tmp_path):
    d = make(tmp_path, dup_k_as_v=False)  # what PEFT exports: no v_proj on k_eq_v layers
    errs = errors(d)
    assert len(errs) == 2 and all("no v_proj copy" in e for e in errs)
    assert at.dup_k_as_v(d, global_layers=GLOBAL) == 4
    assert errors(d) == []
    assert at.dup_k_as_v(d, global_layers=GLOBAL) == 0  # idempotent


def test_dup_k_as_v_casts_fp32_export_to_bf16(tmp_path):
    d = make(tmp_path, dup_k_as_v=False)

    def to_f32(tensors):
        for k, (_, shape, buf) in tensors.items():
            bits = struct.unpack(f"<{len(buf) // 2}H", buf)
            tensors[k] = ("F32", shape, struct.pack(f"<{len(bits)}I", *(b << 16 for b in bits)))

    bf16_before = at.read_safetensors(d / "adapter_model.safetensors")
    edit_tensors(d, to_f32)
    assert "dtype F32" in at.validate(d, real=False, config=TINY, strict=False)["warnings"][0]
    at.dup_k_as_v(d, global_layers=GLOBAL)
    after = at.read_safetensors(d / "adapter_model.safetensors")
    assert all(after[k] == v for k, v in bf16_before.items())  # exact: bf16 -> f32 -> bf16 is lossless
    assert errors(d) == []


def test_v_that_differs_from_k_fails(tmp_path):
    d = make(tmp_path)
    key = f"{at.LORA_PREFIX}.11.self_attn.v_proj.lora_B.weight"
    edit_tensors(d, lambda t: t.__setitem__(key, (t[key][0], t[key][1], bytes(len(t[key][2])))))
    assert any("differs from k_proj" in e for e in errors(d))
    with pytest.raises(at.AdapterError, match="differs from k_proj"):
        at.dup_k_as_v(d, global_layers=GLOBAL)


# ---- broken adapters ----

def test_vision_key_fails(tmp_path):
    d = make(tmp_path)
    key = "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.weight"
    edit_tensors(d, lambda t: t.__setitem__(key, ("BF16", [16, 8], bytes(16 * 8 * 2))))
    assert any("vision" in e for e in errors(d))
    with pytest.raises(at.AdapterError, match="vision"):
        at.validate(d, real=False, config=TINY)


@pytest.mark.parametrize("key", [
    "base_model.model.model.language_model.embed_tokens.lora_A.weight",
    "base_model.model.lm_head.lora_A.weight",
    "base_model.model.model.language_model.layers.0.self_attn.qkv.lora_A.weight",
    "base_model.model.model.language_model.layers.0.mlp.q_proj.lora_A.weight",
])
def test_non_projection_keys_fail(tmp_path, key):
    d = make(tmp_path)
    edit_tensors(d, lambda t: t.__setitem__(key, ("BF16", [16, 64], bytes(16 * 64 * 2))))
    assert errors(d)


@pytest.mark.parametrize("fields, message", [
    ({"rank_pattern": {"q_proj": 8}}, "rank_pattern"),
    ({"alpha_pattern": {"q_proj": 8}}, "alpha_pattern"),
    ({"bias": "all"}, "bias"),
    ({"use_dora": True}, "DoRA"),
    ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    ({"r": 8}, "rank r=8"),
    ({"r": 256}, "1..128"),
])
def test_bad_config_fails(tmp_path, fields, message):
    d = make(tmp_path)
    edit_config(d, **fields)
    assert any(message in e for e in errors(d))


def test_missing_lora_b_and_truncated_file_fail(tmp_path):
    d = make(tmp_path)
    edit_tensors(d, lambda t: t.pop(f"{at.LORA_PREFIX}.2.mlp.gate_proj.lora_B.weight"))
    assert any("lora_A only" in e for e in errors(d))
    st = d / "adapter_model.safetensors"
    st.write_bytes(st.read_bytes()[:-100])
    assert any("truncated" in e for e in errors(d))


def test_real_mode_rejects_wrong_shapes_extra_files_and_bad_names(tmp_path):
    d = make(tmp_path)
    assert any("expected in=5376" in e for e in at.validate(d, real=True, strict=False)["errors"])
    (d / "README.md").write_text("PEFT model card")
    assert any("extra files ['README.md']" in e for e in errors(d, real=True))
    bad = make(tmp_path, name="main lora")
    assert any("adapter name" in e for e in errors(bad, real=True))


def test_real_mode_requires_every_layer(tmp_path):
    d = make(tmp_path)
    assert errors(d, real=True) == []
    edit_tensors(d, lambda t: [t.pop(k) for k in list(t) if ".layers.3.mlp.up_proj." in k])
    assert any("not every layer is covered" in e for e in errors(d, real=True))
    assert errors(d, real=False) == []  # a partial adapter is still loadable


# ---- cross-checks against the real libraries (optional) ----

def test_reference_safetensors_reader_accepts_our_file(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    d = make(tmp_path)
    with safetensors.safe_open(d / "adapter_model.safetensors", framework="numpy") as f:
        assert f.metadata() == {"format": "pt"}
        ours = at.read_safetensors(d / "adapter_model.safetensors")
        assert set(f.keys()) == set(ours)
        for k in f.keys():
            assert f.get_slice(k).get_shape() == ours[k][1] and f.get_slice(k).get_dtype() == "BF16"


def test_scorer_discovers_rank_16(tmp_path):
    adk = pytest.importorskip("adk_submission")
    make(tmp_path / "sub" / "adapters")
    manifest = adk.discover_adapters(str(tmp_path / "sub"), adapter_extensions=frozenset({".safetensors"}))
    assert {n: a.rank for n, a in manifest.adapters.items()} == {"main_lora": 16}
    server = adk.VllmServer(adk.VllmConfig(model="/m", enable_lora=True, max_loras=8, max_lora_rank=128),
                            adapter_manifest=manifest)
    os.unlink(server.log_path)  # the constructor creates a temp log file
    cmd = " ".join(server.build_cmd())
    assert "--enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules main_lora=" in cmd


def _vllm_parser():
    """vLLM's parse_fine_tuned_lora_name and Gemma 4's hf_to_vllm_mapper, executed from the wheel's source.

    vLLM itself cannot be imported here (CUDA, torch), but these pieces are pure Python: pull them out
    of the source files with ast and run them with stub imports.
    """
    if not VLLM_WHEEL.is_file():
        pytest.skip(f"no vLLM wheel at {VLLM_WHEEL} (set VLLM_WHEEL)")
    from collections.abc import Iterable, Mapping
    from dataclasses import dataclass, field
    from typing import Any
    import re

    with zipfile.ZipFile(VLLM_WHEEL) as whl:
        src = {name: whl.read(f"vllm/{name}").decode() for name in
               ("lora/utils.py", "model_executor/models/utils.py", "model_executor/models/gemma4_mm.py")}

    def node(source: str, name: str, inside: str | None = None):
        tree = ast.parse(source)
        if inside:
            tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == inside)
        for n in tree.body:
            targets = [t.id for t in getattr(n, "targets", []) if isinstance(t, ast.Name)]
            if getattr(n, "name", None) == name or name in targets:
                return ast.get_source_segment(source, n)
        raise LookupError(name)

    ns = {"dataclass": dataclass, "field": field, "Mapping": Mapping, "Iterable": Iterable, "Any": Any, "re": re,
          "torch": types.SimpleNamespace(Tensor=object)}
    exec(node(src["model_executor/models/utils.py"], "WeightsMapper"), ns)
    ns["WeightsMapper"] = dataclass(ns["WeightsMapper"])  # the source segment starts after its @dataclass line
    exec(node(src["lora/utils.py"], "parse_fine_tuned_lora_name"), ns)
    gemma = src["model_executor/models/gemma4_mm.py"]
    exec(node(gemma, "hf_to_vllm_mapper", inside="Gemma4ForConditionalGeneration"), ns)
    exec(node(gemma, "packed_modules_mapping", inside="Gemma4ForConditionalGeneration"), ns)
    return ns


def test_key_names_parse_like_vllm(tmp_path):
    ns = _vllm_parser()
    mapper, packed = ns["hf_to_vllm_mapper"], ns["packed_modules_mapping"]
    # Every key of the real 31B layout, plus the tiny dummy file's actual keys.
    keys = [f"{m}.lora_{ab}.weight" for m in at.module_shapes(at.text_config()) for ab in "AB"]
    keys += list(at.read_safetensors(make(tmp_path) / "adapter_model.safetensors"))
    packable = {m for members in packed.values() for m in members} | {"o_proj", "down_proj"}
    for key in keys:
        name, is_a = ns["parse_fine_tuned_lora_name"](key, mapper)
        assert (name, is_a) == at.vllm_module_name(key)
        assert name.startswith("language_model.model.layers.") and name.rsplit(".", 1)[-1] in packable
    # A vision key maps outside the language model, which validate() rejects.
    vision = "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.weight"
    assert ns["parse_fine_tuned_lora_name"](vision, mapper) == at.vllm_module_name(vision)
    assert at.vllm_module_name(vision)[0].startswith("vision_tower.")
    # Our copy of the mapper tables is the wheel's.
    assert dict(mapper.orig_to_new_prefix) == at.HF_TO_VLLM_PREFIX
    assert dict(mapper.orig_to_new_substr) == at.HF_TO_VLLM_SUBSTR
