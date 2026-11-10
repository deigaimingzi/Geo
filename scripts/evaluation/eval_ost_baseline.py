#!/usr/bin/env python3
"""OST-Bench baseline evaluation for compared models.

Supports three model families via their native chat APIs:
  * qwen3_5   - SpatialStack-Qwen3.5-4B (transformers native + VGGT geometry)
  * internvl2 - SenseNova-SI-1.3-InternVL3-8B (custom modeling code)
  * spatiolm  - SpatioLM-Understanding-SenseNovaSI (spatiolm package)

Protocol (official OST-Bench Multi-round):
  * Each session = 8 turns, each turn = 5 images.
  * System prompt from the benchmark is used as the first system message.
  * Each turn appends its 5 NEW images + the turn's official prompt as a user msg.
  * All previous turns' (images + questions + model answers) stay in history.
  * The model generates a JSON response per turn; scoring reuses the official
    OST evaluator (option-LCS / enumeration / MRA + hierarchical macro).

Scoring and data utilities are imported from the GeoMemory-v7 package so that
baseline numbers are directly comparable with GeoMemory's own OST results.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from geomemory.ost import atomic_json, read_sessions, safe_image_path  # noqa: E402
from geomemory.ost_metrics import hierarchical_scores, official_score  # noqa: E402

BENCH = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/OST-Bench"
THRESH = np.arange(0.5, 0.96, 0.05)


# --------------------------------------------------------------------------- #
# Scoring (copied from eval_ost_geomemory.py so this script is self-contained
# for the local scorer; the official scorer is imported above).
# --------------------------------------------------------------------------- #
def _norm(s):
    import re
    return re.sub(r"[\s\"'.,;:!]+", " ", str(s).lower()).strip()


def _is_num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _strip_wrap(x):
    import re
    x = x.strip().strip("[]").strip()
    x = re.sub(r"['\"]\s*,\s*['\"]", ", ", x)
    return x.strip().strip("'\"").strip()


def extract_answer(text):
    t = text.strip()
    if not t:
        return ""
    import re
    m = re.search(r"['\"]?answer['\"]?\s*[:=]\s*", t, re.I)
    if m:
        rest = t[m.end():]
        if rest[:1] in "\"'":
            end = rest.find(rest[0], 1)
            return _strip_wrap(rest[1:end] if end > 0 else rest[1:])
        if rest[:1] == "[":
            end = rest.find("]")
            return _strip_wrap(rest[1:end] if end > 0 else rest[1:])
        return re.split(r"[,}\n]", rest, 1)[0].strip()
    m = re.match(r"\s*\[(.*?)\]", t, re.S)
    if m:
        return _strip_wrap(m.group(1))
    return t


def mra(pred, gt):
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    return float(np.mean([rel < 1 - t for t in THRESH]))


def local_score(rec):
    gt, opts = rec["answer"], rec["option"]
    if rec["type"] == "None" or (gt == "" and not opts):
        return None
    pred = extract_answer(rec["prediction"])
    if opts:
        return float(_norm(pred) == _norm(gt))
    if not _is_num(gt):
        return float(_norm(pred) == _norm(gt))
    import re
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", pred)
    if not nums:
        return 0.0
    v, g = float(nums[0]), float(gt)
    if not np.isfinite(v) or not np.isfinite(g):
        return 0.0
    return mra(v, g) if "(Estimation)" in rec["type"] else float(v == g)


def aggregate(recs, scorer):
    out, by = {}, {}
    total = len(recs)
    recs = [r for r in recs if r["score"] is not None]
    for r in recs:
        by.setdefault(r["type"].split("(")[0], []).append(r["score"])
    for k, v in sorted(by.items()):
        out[k] = float(np.mean(v)) * 100
    out["overall"] = float(np.mean([r["score"] for r in recs])) * 100 if recs else 0.0
    out["num_records"] = len(recs)
    out["num_total_turns"] = total
    out["num_excluded"] = total - len(recs)
    out["macro_category"] = float(np.mean([np.mean(v) for v in by.values()])) * 100 if by else 0.0
    out["category_counts"] = {k: len(v) for k, v in sorted(by.items())}
    types, turns = {}, {}
    for rec in recs:
        types.setdefault(rec["type"], []).append(rec["score"])
        turns.setdefault(str(rec["turn_id"]), []).append(rec["score"])
    out["by_type"] = {k: {"score": float(np.mean(v)) * 100, "n": len(v)} for k, v in sorted(types.items())}
    out["by_turn"] = {k: {"score": float(np.mean(v)) * 100, "n": len(v)} for k, v in sorted(turns.items())}
    return out


def aggregate_with_scorer(recs, scorer_name):
    metrics = aggregate(recs, local_score if scorer_name == "local" else official_score)
    metrics["overall_micro"] = metrics["overall"]
    if scorer_name == "official":
        try:
            metrics["official_dimensions"] = hierarchical_scores(metrics["by_type"])
            metrics["overall"] = metrics["official_dimensions"]["Overall"]
            metrics["official_macro_complete"] = True
        except KeyError as missing:
            metrics["overall"] = None
            metrics["official_macro_complete"] = False
            metrics["official_macro_missing"] = str(missing)
    return metrics


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_frames(paths, root, long_edge=512):
    from PIL import ImageOps
    frames = []
    for p in paths:
        full = safe_image_path(root, p)
        with Image.open(full) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
        w, h = img.size
        scale = long_edge / max(w, h)
        size = tuple(min(long_edge, max(32, round(v * scale / 32) * 32)) for v in (w, h))
        frames.append(img.resize(size, Image.Resampling.BICUBIC))
    return frames


# --------------------------------------------------------------------------- #
# Model wrappers — each exposes .chat_multi_turn(system, turns) -> [str]
# turns = [{"question": str, "images": [PIL], "is_current": bool}, ...]
# --------------------------------------------------------------------------- #
class Qwen35Model:
    """SpatialStack-Qwen3.5-4B via transformers processor + generate."""

    def __init__(self, model_path, vggt_path, device="cuda"):
        from transformers import AutoProcessor, AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        use_geometry = getattr(config, "use_geometry_encoder", False)
        if use_geometry:
            from qwen_vl.model.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGenerationWithGeometry as LoadClass,
            )
        else:
            from transformers import Qwen3_5ForConditionalGeneration as LoadClass
        load_kwargs = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if use_geometry:
            load_kwargs["geometry_encoder_path"] = vggt_path or getattr(config, "geometry_encoder_path", None)
        self.model = LoadClass.from_pretrained(model_path, **load_kwargs).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.device = device
        self.use_geometry = use_geometry
        self.config = config

    @torch.no_grad()
    def chat_multi_turn(self, system_prompt, turns, max_new_tokens=512):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        all_images = []
        for ti, turn in enumerate(turns):
            user_content = []
            for fi, img in enumerate(turn["images"]):
                user_content.append({"type": "text", "text": f"Frame-{fi}: "})
                user_content.append({"type": "image", "image": img})
                all_images.append(img)
            user_content.append({"type": "text", "text": turn["question"]})
            messages.append({"role": "user", "content": user_content})
            if turn.get("answer") is not None:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["answer"]}]})
        if messages and messages[-1]["role"] != "user":
            pass
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=text, images=all_images if all_images else None,
            videos=None, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if self.use_geometry:
            from lmms_eval.models.qwen3_5 import build_qwen3_5_geometry_inputs
            geo_inputs = build_qwen3_5_geometry_inputs(all_images, inputs["image_grid_thw"])
            inputs["geometry_encoder_inputs"] = [torch.stack(geo_inputs).to(self.device)]
        output_ids = self.model.generate(
            **inputs, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, do_sample=False,
            temperature=0, top_p=None, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
        )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return answer.strip()


class InternVL2Model:
    """SenseNova-SI-1.3-InternVL3-8B via native .chat() with history."""

    def __init__(self, model_path, device="cuda"):
        import torchvision.transforms as T
        from transformers import AutoTokenizer, AutoModel
        _patch_internvl_meta_cache(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        self.device = device
        self.input_size = 448
        self.max_num = 1  # tiles per image (keep low for multi-image OST)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(mean=mean, std=std),
        ])
        self.IMG_START = "<img>"
        self.IMG_END = "</img>"
        self.IMG_CONTEXT = "<IMG_CONTEXT>"

    def _to_pixel_values(self, images):
        """Convert PIL images to pixel_values tensor with 1 tile each."""
        pixels = torch.stack([self.transform(img) for img in images]).to(torch.bfloat16).to(self.device)
        return pixels

    @torch.no_grad()
    def chat_multi_turn(self, system_prompt, turns, max_new_tokens=512):
        history = []
        pixel_values = None
        num_patches_list = []
        gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": False,
                   "temperature": 0, "num_beams": 1, "top_p": 1.0}
        for ti, turn in enumerate(turns):
            imgs = turn["images"]
            if imgs:
                pv = self._to_pixel_values(imgs)
                pixel_values = pv if pixel_values is None else torch.cat([pixel_values, pv])
                num_patches_list = num_patches_list + [1] * len(imgs)
                question = "".join(["<image>\n"] * len(imgs)) + turn["question"]
            else:
                question = turn["question"]
            response, history = self.model.chat(
                self.tokenizer, pixel_values, question, gen_cfg,
                history=history if history else None,
                num_patches_list=num_patches_list if num_patches_list else None,
                return_history=True,
            )
        return response.strip()


class SpatioLMModel:
    """SpatioLM-Understanding-SenseNovaSI via spatiolm package."""

    def __init__(self, model_path, device="cuda"):
        import transformers.utils.generic as _g
        if not hasattr(_g, "check_model_inputs"):
            _g.check_model_inputs = lambda fn: fn
        import sys as _sys
        _src = "/tmp/bluecode/repos/spatio-lm/src"
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        _patch_internvl_meta_cache(model_path)
        from spatiolm.models import InternVL3RChatModel
        from transformers import AutoTokenizer
        self.model = InternVL3RChatModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = device
        self.input_size = 448
        import torchvision.transforms as T
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB").resize((self.input_size, self.input_size), Image.BICUBIC)),
            T.ToTensor(), T.Normalize(mean=mean, std=std),
        ])
        self.IMG_START = "<img>"
        self.IMG_END = "</img>"
        self.IMG_CONTEXT = "<IMG_CONTEXT>"

    def _to_pixel_values(self, images):
        return torch.stack([self.transform(img) for img in images]).to(torch.bfloat16).to(self.device)

    @torch.no_grad()
    def chat_multi_turn(self, system_prompt, turns, max_new_tokens=512):
        history = []
        pixel_values = None
        num_patches_list = []
        gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": False,
                   "temperature": 0, "num_beams": 1, "top_p": 1.0}
        for ti, turn in enumerate(turns):
            imgs = turn["images"]
            if imgs:
                pv = self._to_pixel_values(imgs)
                pixel_values = pv if pixel_values is None else torch.cat([pixel_values, pv])
                num_patches_list = num_patches_list + [1] * len(imgs)
                question = "".join(["<image>\n"] * len(imgs)) + turn["question"]
            else:
                question = turn["question"]
            response, history = self.model.chat(
                self.tokenizer, pixel_values, question, gen_cfg,
                history=history if history else None,
                num_patches_list=num_patches_list if num_patches_list else None,
                return_history=True,
            )
        return response.strip()


MODEL_REGISTRY = {
    "qwen3_5": Qwen35Model,
    "internvl2": InternVL2Model,
    "spatiolm": SpatioLMModel,
}


def _patch_internvl_meta_cache(model_path):
    """Monkey-patch torch.linspace to avoid meta-tensor .item() crash.

    transformers >= 5.0 initializes model weights on meta device. InternVL's
    InternVisionEncoder.__init__ calls ``torch.linspace(...).item()`` which
    fails on meta tensors. We temporarily wrap torch.linspace so that if the
    result is a meta tensor, .item() falls back to the raw Python value.

    Also patches PreTrainedModel to add a default all_tied_weights_keys
    attribute for models written against older transformers versions.
    """
    _orig_linspace = torch.linspace

    class _MetaFloat:
        def __init__(self, val):
            self._val = val
        def item(self):
            return self._val
        def __float__(self):
            return float(self._val)

    def _safe_linspace(*args, **kwargs):
        out = _orig_linspace(*args, **kwargs)
        if out.is_meta:
            import numpy as np
            start = float(args[0]) if len(args) > 0 else kwargs.get("start", 0)
            end = float(args[1]) if len(args) > 1 else kwargs.get("end", 0)
            steps = int(args[2]) if len(args) > 2 else kwargs.get("steps", 1)
            vals = list(np.linspace(start, end, steps))
            return [_MetaFloat(v) for v in vals]
        return out

    torch.linspace = _safe_linspace

    # Patch PreTrainedModel.__init__ to set all_tied_weights_keys default
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "_ost_patched"):
        _orig_init = PreTrainedModel.__init__
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
        PreTrainedModel.__init__ = _patched_init
        PreTrainedModel._ost_patched = True


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vggt_path", default="/data/juicefs_sharing_data/11188211/Qinjia-2026/vision_infer/depth-VLM/depth-model/VGGT-1B")
    ap.add_argument("--bench", default=f"{BENCH}/OST_bench_eval_nested.json")
    ap.add_argument("--image_root", default=f"{BENCH}/image_upload")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--long_edge", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--scorer", choices=("local", "official"), default="official")
    a = ap.parse_args()
    if not 0 <= a.rank < a.world:
        ap.error("rank must be within world size")
    os.makedirs(a.out_dir, exist_ok=True)
    data = read_sessions(a.bench)
    if a.limit:
        data = data[:a.limit]
    expected = {(s["scan_id"], int(t["turn_id"])) for s in data for t in s["user_message"]}
    scorer = official_score if a.scorer == "official" else local_score

    protocol = {
        "protocol": "ost-baseline-multi-round-v1",
        "model": a.model, "model_path": str(Path(a.model_path).resolve()),
        "bench": str(Path(a.bench).resolve()), "limit": a.limit, "world": a.world,
        "max_new_tokens": a.max_new_tokens, "long_edge": a.long_edge,
        "dialogue_history": "all turns: images+questions+model responses",
        "scoring": ("official option-LCS/enumeration/MRA + hierarchical macro" if a.scorer == "official"
                    else "local type-aware exact-match/MRA(0.50:0.05:0.95)"),
    }
    protocol_path = Path(a.out_dir) / "eval_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise ValueError("Output directory belongs to a different evaluation protocol")
    elif a.rank == 0:
        atomic_json(protocol_path, protocol)

    if a.merge:
        recs = []
        for rank in range(a.world):
            with open(os.path.join(a.out_dir, f"predictions_rank{rank}.json")) as handle:
                recs += json.load(handle)
        keys = [(r["scan_id"], int(r["turn_id"])) for r in recs]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f"Incomplete/duplicate evaluation: records={len(keys)}, expected={len(expected)}")
        for rec in recs:
            rec["score"] = scorer(rec)
        m = aggregate_with_scorer(recs, a.scorer)
        if a.scorer == "official" and not a.limit and not m.get("official_macro_complete"):
            raise ValueError("Full official evaluation is missing one or more task types")
        m["coverage_complete"] = True
        atomic_json(Path(a.out_dir) / "metrics.json", m)
        print(json.dumps(m, indent=2))
        return

    # Set device for this rank — each process sees only its GPU via CUDA_VISIBLE_DEVICES
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Add SOP-QV-MH-ablation src to path for qwen3_5 model class
    if a.model == "qwen3_5":
        sop_src = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/src"
        if sop_src not in sys.path:
            sys.path.insert(0, sop_src)

    kwargs = {"device": device}
    if a.model == "qwen3_5":
        kwargs["vggt_path"] = a.vggt_path
    model = MODEL_REGISTRY[a.model](a.model_path, **kwargs)

    data = data[a.rank::a.world]
    out_path = os.path.join(a.out_dir, f"predictions_rank{a.rank}.json")
    recs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["scan_id"], int(r["turn_id"])) for r in recs}

    for si, s in enumerate(tqdm(data, desc=f"rank{a.rank}", dynamic_ncols=True)):
        sid = s["scan_id"]
        if all((sid, t["turn_id"]) in done for t in s["user_message"]):
            continue
        system_prompt = s.get("system_prompt", "")
        # Build multi-turn chat: all turns with images + questions + prior answers
        chat_turns = []
        prev_answers = {}
        for t in s["user_message"]:
            tid = t["turn_id"]
            if (sid, tid) in done:
                prev_pred = next((r["prediction"] for r in recs if r["scan_id"] == sid and int(r["turn_id"]) == int(tid)), "")
                chat_turns.append({"question": t["prompt"], "images": None, "answer": prev_pred})
                continue
            # Current turn: load images for this turn
            frames = load_frames(t["image_paths"], a.image_root, a.long_edge)
            chat_turns.append({"question": t["prompt"], "images": frames, "answer": None, "is_current": True})
            # Run inference on all turns up to and including this one
            pred = model.chat_multi_turn(system_prompt, chat_turns, max_new_tokens=a.max_new_tokens)
            chat_turns[-1]["answer"] = pred
            r = {"session": sid, "scan_id": sid, "turn_id": tid, "type": t["type"],
                 "option": t["option"], "answer": str(t["answer"]), "prediction": pred}
            r["score"] = scorer(r)
            recs.append(r)
            done.add((sid, int(tid)))
            atomic_json(Path(out_path), recs)
        atomic_json(Path(out_path), recs)

    m = aggregate_with_scorer(recs, a.scorer)
    atomic_json(Path(out_path), recs)
    atomic_json(Path(a.out_dir) / f"metrics_rank{a.rank}.json", m)
    print(json.dumps(m))


if __name__ == "__main__":
    main()
