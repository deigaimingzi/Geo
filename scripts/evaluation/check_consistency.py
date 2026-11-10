#!/usr/bin/env python3
"""Consistency checks for the GeoMemory-v3 VSI eval path.
A) protocol: eval prompt ids == training collator prompt ids (token-level); pad count == feature count
B) cache-free oracle: full-sequence recompute (positions from get_rope_index on the whole sequence,
   no KV cache) vs cached step decode: logits + tokens, teacher-forced 16 steps
C) HF generate(inputs_embeds=...) greedy tokens vs manual decode
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vsi_geomemory import *  # noqa
from eval_vsi_geomemory import prepare, greedy_decode, load_all
from eval_vsi_mra import VSIBenchDataset

CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = torch.device("cuda:0")
model, processor, builder = load_all(CK, DEFAULT_QWEN, dev)
tok = builder.tokenizer
ds = VSIBenchDataset(f"{VSI}/test.jsonl", VSI, 32, None, 0, 1)
step = len(ds) // N
STEPS = 16
for k in range(N):
    item = ds[k * step]
    P = prepare(model, builder, processor, item["frames"], item["question"], dev)
    ids = P["ids"]; n_pad = int(P["mm"].sum())
    print(f"\n=== sample {k} [{item['question_type']}] frames={len(item['frames'])} prompt_len={ids.shape[1]} pads={n_pad} grid0={P['grid'][0].tolist()}")
    # A: protocol
    tail = tok.decode(ids[0, -12:])
    print("A tail:", repr(tail))
    assert tail.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "assistant prefix differs from training"
    per = [int(x) for x in (P["grid"].prod(-1) // 4).tolist()]
    assert n_pad == sum(per), (n_pad, sum(per))
    ids_ref, _ = builder.build(item["question"], "x", per)   # training path (answer appended)
    assert ids_ref[: ids.shape[1]] == ids[0].tolist(), "prompt ids differ from training build()"
    # B: cached vs cache-free oracle, teacher forced on cached greedy tokens
    toks, lg = greedy_decode(model, builder, P, STEPS, return_logits=True, forced=None)
    forced = toks + [builder.im_end_id] * (STEPS - len(toks))
    toks_f, lg_f = greedy_decode(model, builder, P, STEPS, return_logits=True, forced=forced)
    dec = model._qwen_decoder(); lm = model.language_model; embed = lm.get_input_embeddings()
    worst, mism = 0.0, 0
    for t in range(len(toks) + 1):
        seq = torch.cat([ids, torch.tensor([forced[:t]], device=dev, dtype=ids.dtype)], 1) if t else ids
        mm = (seq == builder.image_pad_id)
        e = P["emb"] if t == 0 else torch.cat([P["emb"], embed(seq[:, ids.shape[1]:])], 1)
        pos, _ = dec.get_rope_index(seq, mm_token_type_ids=mm.to(torch.int32), image_grid_thw=P["grid"],
                                    attention_mask=torch.ones_like(seq))
        h = dec(inputs_embeds=e, position_ids=pos, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        ref = lm.lm_head(h).float().cpu()
        d = (ref - lg_f[t]).abs().max().item(); worst = max(worst, d)
        mism += int(ref.argmax(-1).item() != lg_f[t].argmax(-1).item())
        # position continuity check for decode step
        if t > 0:
            assert int(pos[0, 0, -1]) == int(P["pos"].max()) + t, (int(pos[0,0,-1]), int(P["pos"].max()) + t)
    print(f"B cached-vs-nocache: steps={len(toks)+1} max|dlogit|={worst:.4f} argmax_mismatch={mism}")
    # C: HF generate
    try:
        g = lm.generate(inputs_embeds=P["emb"], attention_mask=torch.ones_like(ids), max_new_tokens=STEPS,
                        do_sample=False, num_beams=1, use_cache=True, mm_token_type_ids=P["mm"].to(torch.int32),
                        image_grid_thw=P["grid"], input_ids=ids) if False else \
            lm.generate(inputs_embeds=P["emb"], attention_mask=torch.ones_like(ids), max_new_tokens=STEPS,
                        do_sample=False, num_beams=1, use_cache=True)
        gt = g[0].tolist()
        gt = gt[: gt.index(builder.im_end_id)] if builder.im_end_id in gt else gt
        print("C generate:", gt, "| manual:", toks, "| equal:", gt == toks)
    except Exception as ex:
        print("C generate failed:", type(ex).__name__, str(ex)[:200])
    print("manual text:", repr(tok.decode(toks)), "| gt:", item["answer"])
