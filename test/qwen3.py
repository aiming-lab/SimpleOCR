from transformers import AutoModelForCausalLM, AutoTokenizer
from .interceptor import QKVInterceptorQwen3Eager

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
inv_freq = model.model.rotary_emb.inv_freq  # shape [head_dim/2]

# 2. 打印前后几项，看看是递增还是递减
print("First 10 inv_freq:", inv_freq[:10])
print("Last 10 inv_freq:", inv_freq[-10:])
# prepare the model input
prompt = '''
Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
'''
# prompt = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

import torch
# Stage 2: add small Gaussian noise to Q (post-norm, pre-RoPE)
# def stage2_noise(layer_idx, q, k, v):
#     noise = torch.randn_like(q) * 0.01
#     return q + noise, k, v

# Stage 3: scale the last quarter of heads (toy demo)
# def stage3_scale(layer_idx, q, k, v):
#     h = q.size(1)
#     cut = int(0.75 * h)
#     q[:, cut:, :, :] *= 0
#     k[:, cut:, :, :] *= 0
#     return q, k, v

# 10: 1648
# 2: 2156
# 16: err Wait, arctan(3/0. Wait, arctan(3/0. Wait, arctan(3/0.
# 20: err
# 120: err
# 122: 1751
# 124：2262
# 126: 2482
# 118: err

def on_stage3_fn(layer_idx, q, k, v):
    head_dim = q.size(-1)
    start = head_dim // 2
    # q[..., head_dim-10:] = 0.0
    q[..., :8] = 0.0
    return q, k, v

interceptor = QKVInterceptorQwen3Eager(
    model=model.model,             # IMPORTANT: pass the base model (e.g., Qwen3ForCausalLM.model)
    on_stage2=None,        # point (2)
    on_stage3=on_stage3_fn,        # point (3)
    save_tensors=False,
    verbose=True,
)

with interceptor:
    # outputs = model(**model_inputs)   # preserves eager path & caching behavior
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=12000,
        use_cache=True,
        do_sample=False
    )
# Inspect captured tensors
# print(len(interceptor.captures), interceptor.captures[0]["stage"], interceptor.captures[0]["q"].shape)
# import pdb; pdb.set_trace()
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print(len(output_ids))

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=False).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
