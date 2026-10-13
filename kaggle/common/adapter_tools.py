"""Make, fix and check LoRA adapters for the served Gemma 4 31B (the organizers' patched vLLM 0.19.1).

Used locally before every upload, in the FT-0 notebook (dummy adapter) and in the training notebook
(export). Kaggle runs one file, so scripts/kaggle_build.py pastes this module in place of
`from adapter_tools import ...`: keep it free of `from __future__` lines, repo imports and an
`if __name__ == "__main__"` block (it would run inside the notebook).

Only the standard library is required. numpy (always present on Kaggle) is used when installed, to
make and cast the 31B-sized tensors quickly. Safetensors files are read and written directly; the
format is simple (8-byte little-endian header length, a JSON header {name: {dtype, shape,
data_offsets}}, then the raw bytes back to back), and doing it by hand handles bf16 without torch
(numpy has no bf16 type).

The layout vLLM accepts (see the plan's adapter_spec):
  keys    base_model.model.model.language_model.layers.N.{self_attn,mlp}.X_proj.lora_{A,B}.weight
  shapes  lora_A [r, in], lora_B [out, r], bf16
vLLM strips 'base_model.model.', then gemma4_mm's hf_to_vllm_mapper turns 'model.language_model.'
into 'language_model.model.'; q/k/v are packed into qkv_proj and gate/up into gate_up_proj.

Why dup_k_as_v exists: on the 10 full-attention layers Gemma 4 31B has no v_proj
(attention_k_eq_v): HF computes V = k_proj(x), so a k_proj LoRA changes V too. vLLM instead copies
the base k_proj weights into the V slot of qkv_proj but only applies a V-slot LoRA from a v_proj
entry. Writing the k_proj LoRA again as v_proj makes vLLM compute the same V as training did.
"""

import json
import math
import os
import random
import re
import struct
from array import array
from pathlib import Path

try:
    import numpy as np
except ImportError:  # local CPU tests; Kaggle images always have numpy
    np = None

LORA_PREFIX = "base_model.model.model.language_model.layers"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
# PEFT target regex. The plain 7-name list would also hit the vision tower (PEFT then raises on
# Gemma4ClippableLinear), and 'all-linear' adds vision/embedding modules that break vLLM startup.
TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
ATTN_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"
VLLM_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)  # --max-lora-rank buckets (adk server.py:58-67)
MAX_RANK = 128  # competition limit
FORBIDDEN = ("vision", "audio", "embed", "lm_head")
ADAPTER_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")  # adk_submission rejects other adapter names
ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors"}

# Copied from the patched wheel (vllm/model_executor/models/gemma4_mm.py hf_to_vllm_mapper). vLLM
# applies every matching rule in order; it does not stop at the first match.
HF_TO_VLLM_SUBSTR = {".moe.experts.gate_up_proj": ".moe.gate_up_proj", ".moe.experts.down_proj": ".moe.down_proj"}
HF_TO_VLLM_PREFIX = {
    "model.embed_audio.": "embed_audio.",
    "model.embed_vision.": "embed_vision.",
    "model.language_model.": "language_model.model.",
    "model.vision_tower.": "vision_tower.",
    "model.audio_tower.": "audio_tower.",
    "lm_head.": "language_model.lm_head.",
    "model": "language_model.model",
}
# Module suffixes vLLM accepts in a LoRA for this model (supported LinearBase modules with
# packed_modules_mapping expanded, worker_manager._load_adapter). Anything else: ValueError at startup.
VLLM_EXPECTED_SUFFIXES = set(ATTN + MLP) | {"embedding_projection"}

# text_config of the served model (google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2, config.json).
GEMMA4_31B = {
    "hidden_size": 5376, "intermediate_size": 21504, "num_hidden_layers": 60,
    "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
    "num_global_key_value_heads": 4, "global_head_dim": 512, "attention_k_eq_v": True,
    "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)],
}


class AdapterError(ValueError):
    pass


# ---- model geometry ----

def text_config(config: dict | str | Path | None = None) -> dict:
    """config: None (the served 31B), a config.json path, a full config dict or its text_config."""
    if config is None:
        return dict(GEMMA4_31B)
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    return dict(config.get("text_config", config))


def k_eq_v_layers(tc: dict) -> set[int]:
    """Layers without a v_proj (V = K): the full-attention ones when attention_k_eq_v is set."""
    if not tc.get("attention_k_eq_v"):
        return set()
    return {i for i, t in enumerate(tc["layer_types"]) if t == "full_attention"}


def module_shapes(tc: dict, attn_only: bool = False, v_on_k_eq_v: bool = True) -> dict[str, tuple[int, int]]:
    """PEFT module name -> (in_features, out_features) for every module an adapter should have.

    v_on_k_eq_v=False lists only what PEFT trains (k_eq_v layers have no v_proj module to train).
    """
    H, I = tc["hidden_size"], tc["intermediate_size"]
    shared = k_eq_v_layers(tc)
    shapes = {}
    for i, layer_type in enumerate(tc["layer_types"]):
        full = layer_type == "full_attention"
        head_dim = tc["global_head_dim"] if full else tc["head_dim"]
        kv_heads = tc["num_global_key_value_heads"] if i in shared else tc["num_key_value_heads"]
        q, kv = tc["num_attention_heads"] * head_dim, kv_heads * head_dim
        p = f"{LORA_PREFIX}.{i}"
        shapes[f"{p}.self_attn.q_proj"] = (H, q)
        shapes[f"{p}.self_attn.k_proj"] = (H, kv)
        if i not in shared or v_on_k_eq_v:
            shapes[f"{p}.self_attn.v_proj"] = (H, kv)
        shapes[f"{p}.self_attn.o_proj"] = (q, H)
        if not attn_only:
            shapes[f"{p}.mlp.gate_proj"] = (H, I)
            shapes[f"{p}.mlp.up_proj"] = (H, I)
            shapes[f"{p}.mlp.down_proj"] = (I, H)
    return shapes


def adapter_config(rank: int, alpha: int, attn_only: bool = False) -> dict:
    """adapter_config.json with the fields vLLM's PEFTHelper reads set to values it accepts."""
    return {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "google/gemma-4-31b-it",
        "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
        "use_dora": False, "use_rslora": False, "modules_to_save": None,
        "target_modules": ATTN_REGEX if attn_only else TARGET_REGEX,
        "rank_pattern": {}, "alpha_pattern": {}, "inference_mode": True, "fan_in_fan_out": False,
    }


# ---- safetensors I/O (tensors kept as (dtype, shape, raw bytes)) ----

Tensors = dict[str, tuple[str, list[int], bytes]]
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}


def read_header(path: Path) -> tuple[dict, int]:
    """(header without __metadata__, byte offset where the tensor data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def read_safetensors(path: str | Path) -> Tensors:
    """Read every tensor; fail on a truncated or inconsistent file (e.g. a partial upload)."""
    path = Path(path)
    header, start = read_header(path)
    data = path.read_bytes()[start:]
    tensors, end = {}, 0
    for name, m in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = m["data_offsets"]
        if a != end or b - a != math.prod(m["shape"]) * DTYPE_BYTES.get(m["dtype"], 0):
            raise AdapterError(f"{path}: tensor {name} has bad data_offsets {m['data_offsets']}")
        tensors[name] = (m["dtype"], list(m["shape"]), data[a:b])
        end = b
    if end != len(data):
        raise AdapterError(f"{path}: header covers {end} data bytes but the file has {len(data)} (truncated?)")
    return tensors


def write_safetensors(path: str | Path, tensors: Tensors) -> None:
    """Write tensors sorted by name, with the {"format": "pt"} metadata PEFT writes."""
    path = Path(path)
    header, offset = {}, 0
    for name in sorted(tensors):
        dtype, shape, buf = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(buf)]}
        offset += len(buf)
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # the reference writer pads the header to 8 bytes
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in sorted(tensors):
            f.write(tensors[name][2])
    os.replace(tmp, path)


def _floats_to_bf16(values) -> bytes:
    """bf16 = the top 16 bits of a float32, rounded to nearest even (pure Python fallback)."""
    u = array("I", array("f", values).tobytes())
    return array("H", (((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF for x in u)).tobytes()


def to_bf16(dtype: str, buf: bytes) -> bytes:
    if dtype == "BF16":
        return buf
    if dtype not in ("F32", "F16"):
        raise AdapterError(f"cannot cast {dtype} to bf16")
    if np is not None:
        u = np.frombuffer(buf, np.float32 if dtype == "F32" else np.float16).astype(np.float32).view(np.uint32)
        return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    return _floats_to_bf16(struct.unpack(f"<{len(buf) // DTYPE_BYTES[dtype]}{'f' if dtype == 'F32' else 'e'}", buf))


def _random_bf16(rng, n: int, std: float) -> bytes:
    if np is not None:
        return to_bf16("F32", (rng.standard_normal(n, dtype=np.float32) * np.float32(std)).tobytes())
    return _floats_to_bf16([rng.gauss(0.0, std) for _ in range(n)])


# ---- make / fix ----

K_LORA_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.k_proj\.lora_([AB])\.weight$")


def _add_v_duplicates(tensors: Tensors, layers: set[int]) -> int:
    """Copy each k_proj LoRA on `layers` to v_proj. Idempotent; refuses to overwrite a different v_proj."""
    added = 0
    for name in sorted(tensors):
        m = K_LORA_RE.search(name)
        if not (m and int(m.group(1)) in layers):
            continue
        v_name = name.replace(".k_proj.", ".v_proj.")
        if v_name in tensors:
            if tensors[v_name] != tensors[name]:
                raise AdapterError(f"{v_name} exists and differs from k_proj (trained v_proj on a k_eq_v layer?)")
            continue
        tensors[v_name] = tensors[name]
        added += 1
    return added


def make_dummy(out: str | Path, rank: int = 16, dup_k_as_v: bool = True, attn_only: bool = False,
               config: dict | str | Path | None = None, *, alpha: int | None = None, b_std: float = 0.02,
               seed: int = 0) -> Path:
    """Write a PEFT-format LoRA with random weights for the model in `config`, without loading the model.

    lora_A ~ N(0, 1/in) and lora_B ~ N(0, b_std^2). Both are nonzero on purpose: PEFT starts B at
    zero, which makes a fresh adapter a no-op and hides whether vLLM really applied it.
    alpha defaults to 2*rank (scaling 2, as in training).
    """
    tc = text_config(config)
    rng = np.random.default_rng(seed) if np is not None else random.Random(seed)
    tensors: Tensors = {}
    for name, (fan_in, fan_out) in module_shapes(tc, attn_only, v_on_k_eq_v=False).items():
        tensors[f"{name}.lora_A.weight"] = ("BF16", [rank, fan_in], _random_bf16(rng, rank * fan_in, fan_in ** -0.5))
        tensors[f"{name}.lora_B.weight"] = ("BF16", [fan_out, rank], _random_bf16(rng, fan_out * rank, b_std))
    if dup_k_as_v:
        _add_v_duplicates(tensors, k_eq_v_layers(tc))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(adapter_config(rank, alpha or 2 * rank, attn_only), indent=2))
    write_safetensors(out / "adapter_model.safetensors", tensors)
    return out


def dup_k_as_v(adapter_dir: str | Path, global_layers=range(5, 60, 6)) -> int:
    """Add a v_proj copy of the k_proj LoRA on the k_eq_v layers; also casts every tensor to bf16.

    Run on PEFT's export (which may be fp32). Returns how many tensors were added (20 for the 31B
    with k_proj on all 10 global layers, 0 when run a second time).
    """
    path = Path(adapter_dir) / "adapter_model.safetensors"
    tensors = {k: ("BF16", shape, to_bf16(dtype, buf)) for k, (dtype, shape, buf) in read_safetensors(path).items()}
    added = _add_v_duplicates(tensors, set(global_layers))
    write_safetensors(path, tensors)
    return added


# ---- validate ----

def vllm_module_name(key: str) -> tuple[str, bool]:
    """Replica of vllm/lora/utils.py parse_fine_tuned_lora_name with gemma4_mm's weights mapper.

    Returns (vLLM module name, is_lora_A). tests/test_adapter_tools.py checks it against the wheel's code.
    """
    def map_name(k: str) -> str:
        for old, new in HF_TO_VLLM_SUBSTR.items():
            if old in k:
                k = k.replace(old, new, 1)
        for old, new in HF_TO_VLLM_PREFIX.items():
            if k.startswith(old):
                k = k.replace(old, new, 1)
        return k

    if key.startswith("base_model.model."):
        key = "base_model.model." + map_name(key.replace("base_model.model.", ""))
    else:
        key = map_name(key)
    start = 2 if key.startswith("base_model.model.") else 0
    parts = key.split(".")
    if parts[-1] == "weight" and parts[-2] in ("lora_A", "lora_B"):
        return ".".join(parts[start:-2]), parts[-2] == "lora_A"
    if parts[-1] in ("lora_embedding_A", "lora_embedding_B"):
        return ".".join(parts[start:-1]), parts[-1] == "lora_embedding_A"
    raise AdapterError(f"{key} is unsupported LoRA weight")


VLLM_MODULE_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")


def validate(adapter_dir: str | Path, real: bool = True, config: dict | str | Path | None = None,
             strict: bool = True) -> dict:
    """Check an adapter dir against what vLLM, adk_submission and our training layout require.

    Always: PEFTHelper's config rules, every key parses to a language-model q/k/v/o/gate/up/down
    module, no vision/audio/embedding/lm_head keys, A and B ranks match r, and on k_eq_v layers the
    v_proj LoRA exists and is a byte copy of k_proj's.
    Shapes are checked against `config` (the 31B when real=True and config is None).
    real=True (the files we ship) also requires: only the two adapter files, a valid adapter name,
    bf16 tensors, and every layer covered (all 7 projections, or the attention-only fallback).
    Returns a report; with strict=True raises AdapterError listing every problem.
    """
    d = Path(adapter_dir)
    errors, warnings = [], []
    report = {"dir": str(d), "real": real}

    def done() -> dict:
        report.update(ok=not errors, errors=errors, warnings=warnings)
        if errors and strict:
            raise AdapterError(f"{d}: adapter check failed:\n  " + "\n  ".join(errors))
        return report

    missing = sorted(f for f in ADAPTER_FILES if not (d / f).is_file())
    if missing:
        errors.append(f"missing {missing}")
        return done()
    if real:
        extra = sorted(p.name for p in d.iterdir() if p.name not in ADAPTER_FILES)
        if extra:  # a stray .safetensors becomes a second adapter; anything else is dead weight in the zip
            errors.append(f"extra files {extra}: ship only {sorted(ADAPTER_FILES)}")
        if not ADAPTER_NAME_RE.match(d.name):
            errors.append(f"adapter name {d.name!r} must match {ADAPTER_NAME_RE.pattern}")

    # adapter_config.json, as vLLM's PEFTHelper.from_dict + validate_legal read it.
    cfg = json.loads((d / "adapter_config.json").read_text())
    for field in ("r", "lora_alpha", "target_modules"):
        if field not in cfg:
            errors.append(f"adapter_config.json lacks required field {field!r}")
    r = cfg.get("r")
    if not (isinstance(r, int) and 0 < r <= MAX_RANK):
        errors.append(f"r={r!r}: must be an int in 1..{MAX_RANK}")
        return done()
    if cfg.get("peft_type", "LORA") != "LORA":
        errors.append(f"peft_type {cfg['peft_type']!r} is not LORA")
    if cfg.get("bias", "none") != "none":
        errors.append(f"bias={cfg['bias']!r}: vLLM requires 'none'")
    if cfg.get("use_dora"):
        errors.append("use_dora=true: vLLM does not support DoRA")
    if cfg.get("modules_to_save"):
        errors.append(f"modules_to_save={cfg['modules_to_save']!r}: vLLM requires null")
    for field in ("rank_pattern", "alpha_pattern"):
        if cfg.get(field):  # vLLM uses one r and lora_alpha for every module: per-module values are ignored
            errors.append(f"{field} must be empty (vLLM ignores it, so outputs would be mis-scaled)")
    alpha = cfg.get("lora_alpha", 0)
    report.update(r=r, lora_alpha=alpha, max_lora_rank=next(b for b in VLLM_RANKS if b >= r),
                  scaling=alpha / (math.sqrt(r) if cfg.get("use_rslora") else r))

    # adapter_model.safetensors
    try:
        tensors = read_safetensors(d / "adapter_model.safetensors")
    except (AdapterError, ValueError, struct.error) as e:
        errors.append(str(e))
        return done()
    modules: dict[str, dict[str, str]] = {}  # PEFT module name -> {"A": key, "B": key}
    for key, (dtype, shape, _) in tensors.items():
        if any(word in key for word in FORBIDDEN):
            errors.append(f"{key}: vision/audio/embedding/lm_head keys break vLLM startup")
            continue
        if dtype != "BF16":
            (errors if real else warnings).append(f"{key}: dtype {dtype}, expected BF16")
        try:
            vllm_name, is_a = vllm_module_name(key)
        except AdapterError as e:
            errors.append(str(e))
            continue
        m = VLLM_MODULE_RE.match(vllm_name)
        if vllm_name.rsplit(".", 1)[-1] not in VLLM_EXPECTED_SUFFIXES:
            errors.append(f"{key} -> {vllm_name}: not a module vLLM can load a LoRA into (ValueError at startup)")
        elif not m or m.group(3) not in (ATTN if m.group(2) == "self_attn" else MLP):
            errors.append(f"{key} -> {vllm_name}: not a language-model attention/MLP projection")
        elif len(shape) != 2 or shape[0 if is_a else 1] != r:
            errors.append(f"{key}: shape {shape} does not have rank r={r} on the {'first' if is_a else 'second'} axis")
        else:
            modules.setdefault(key.rsplit(".", 2)[0], {})["A" if is_a else "B"] = key
    for module, ab in sorted(modules.items()):
        if set(ab) != {"A", "B"}:
            errors.append(f"{module}: has lora_{'A' if 'A' in ab else 'B'} only")

    # Shapes, coverage and the k_eq_v duplicate.
    tc = text_config(config) if (real or config is not None) else None
    if tc is not None:
        layers = k_eq_v_layers(tc)
        expected = module_shapes(tc)
        for module, ab in sorted(modules.items()):
            if module not in expected:
                errors.append(f"{module}: not in the model (layer out of range?)")
            elif set(ab) == {"A", "B"}:
                fan_in, fan_out = expected[module]
                a_shape, b_shape = tensors[ab["A"]][1], tensors[ab["B"]][1]
                if a_shape[1] != fan_in or b_shape[0] != fan_out:
                    errors.append(f"{module}: A{a_shape} B{b_shape}, expected in={fan_in} out={fan_out}")
        if real:
            complete = [set(module_shapes(tc, attn_only=a)) for a in (False, True)]
            if set(modules) not in complete:
                lacking = sorted(complete[0] - set(modules))
                errors.append(f"{len(modules)} modules: not every layer is covered, e.g. missing {lacking[:3]} "
                              f"(expected {len(complete[0])}, or {len(complete[1])} attention-only)")
    else:  # no geometry given: check the duplicate on the 31B's global layers
        layers = k_eq_v_layers(GEMMA4_31B)
    for layer in sorted(layers):
        k = f"{LORA_PREFIX}.{layer}.self_attn.k_proj"
        v = f"{LORA_PREFIX}.{layer}.self_attn.v_proj"
        if v in modules and k not in modules:
            errors.append(f"{v}: v_proj LoRA on k_eq_v layer {layer} without k_proj")
        elif k in modules and v not in modules:
            errors.append(f"{k}: k_eq_v layer {layer} has no v_proj copy; run dup_k_as_v "
                          f"(vLLM would apply the k_proj delta to K only, training applied it to V too)")
        elif k in modules:
            for ab in ("A", "B"):
                kt, vt = tensors.get(f"{k}.lora_{ab}.weight"), tensors.get(f"{v}.lora_{ab}.weight")
                if kt != vt:
                    errors.append(f"{v}.lora_{ab}: differs from k_proj's (k_eq_v duplicate must be an exact copy)")

    report.update(n_tensors=len(tensors), n_modules=len(modules),
                  size_mib=round((d / "adapter_model.safetensors").stat().st_size / 2**20, 1),
                  targets=sorted({m.rsplit(".", 1)[-1] for m in modules}))
    return done()

