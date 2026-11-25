#!/usr/bin/env python3
"""CPU reference for vllm.sh and sglang.sh: greedy generation with transformers.

Writes a JSON file of {prompt, text, token_ids} records.
Environment:
  MODEL       path to the model (required)
  DTYPE       float16, bfloat16 or float32 (default float32)
  MAX_TOKENS  maximum new tokens (default 50)
  PENALTY     the repetition penalty: a number (default 1.0: plain greedy, what vLLM's and SGLang's
              engines do when asked for a temperature of 0), or `model` for what the model's own
              generation_config.json asks (1.1 for Qwen2.5), which transformers applies by default
              and vLLM's HTTP server too
  OUTFILE     output JSON path (required)
"""
import os, sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get('MODEL')
if not MODEL:
    print('MODEL is required', file=sys.stderr); sys.exit(2)
DTYPE = os.environ.get('DTYPE', 'float32')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '50'))
PENALTY = os.environ.get('PENALTY', '1.0')
OUTFILE = os.environ.get('OUTFILE')
if not OUTFILE:
    print('OUTFILE is required', file=sys.stderr); sys.exit(2)

torch_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(DTYPE, torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch_dtype)
model = model.to('cpu')
model.eval()

PROMPTS = [
    "The capital of France is",
    "What is 2 + 2? The answer is",
    "Explain briefly what a GPU is:",
]
results = []
for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        # (The model's own generation_config asks for a repetition penalty of 1.1, which
        # transformers applies here unless told otherwise, and which vLLM's and SGLang's engines
        # do not apply: a reference made with it left their text at the 3rd, 5th and 22nd token,
        # which was first put down to float16.)
        penalty = {} if PENALTY == 'model' else {'repetition_penalty': float(PENALTY)}
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                 temperature=None, top_p=None, output_scores=True,
                                 return_dict_in_generate=True, **penalty)
    new_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    # By how much each token led the next most likely one (after the penalty, if any): where
    # that is small, a computation in float16 may choose the other (greedy_compare.py).
    margins = []
    for scores in outputs.scores:
        top = torch.topk(scores[0].double(), 2).values
        margins.append(round((top[0] - top[1]).item(), 4))
    results.append({
        'prompt': prompt,
        'text': tokenizer.decode(new_tokens, skip_special_tokens=True),
        'token_ids': new_tokens.tolist(),
        'token_texts': [tokenizer.decode([t], skip_special_tokens=True) for t in new_tokens.tolist()],
        'margins': margins,
        'n_tokens': len(new_tokens),
    })
    print(f'  {prompt!r} -> {len(new_tokens)} tokens')
with open(OUTFILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'wrote {OUTFILE}')
