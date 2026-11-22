#!/usr/bin/env python3
"""GPU inference with transformers (the same operations vLLM uses).

Runs greedy generation on GPU and compares token by token with a CPU reference.
Environment:
  MODEL       path to the model (required)
  DTYPE       float16 or float32 (default float32)
  MAX_TOKENS  maximum new tokens (default 50)
  REF_FILE    JSON reference from vllm-ref.py (required)
Prints a line for each prompt (MATCH, or the first divergence), then `verdict: ALL_MATCH` or
`verdict: SOME_DIFFER`.
"""
import os, sys, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ['MODEL']
DTYPE = os.environ.get('DTYPE', 'float32')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '50'))
REF_FILE = os.environ['REF_FILE']

torch_dtype = torch.float16 if DTYPE == 'float16' else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch_dtype)
model = model.to('cuda')
model.eval()

ref = json.load(open(REF_FILE))
all_match = True
total_tokens = 0
total_time = 0.0

for i, r in enumerate(ref):
    prompt = r['prompt']
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        # (Plain greedy, as the reference of vllm-ref.py: no repetition penalty.)
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                 temperature=None, top_p=None, repetition_penalty=1.0)
    torch.cuda.synchronize()
    t1 = time.time()
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:].tolist()
    n = len(new_tokens)
    total_tokens += n
    total_time += (t1 - t0)

    ref_ids = r['token_ids']
    if new_tokens == ref_ids:
        print(f'  prompt {i}: MATCH ({n} tokens, {t1-t0:.2f}s)')
    else:
        all_match = False
        for j, (rt, gt) in enumerate(zip(ref_ids, new_tokens)):
            if rt != gt:
                print(f'  prompt {i}: DIFFER at token {j}: ref={rt} gpu={gt}')
                break
        else:
            print(f'  prompt {i}: lengths differ: ref={len(ref_ids)} gpu={n}')

if total_time > 0:
    print(f'  {total_tokens} tokens in {total_time:.2f}s = {total_tokens/total_time:.1f} tok/s')
if all_match:
    print('verdict: ALL_MATCH')
else:
    print('verdict: SOME_DIFFER')
sys.exit(0 if all_match else 1)
