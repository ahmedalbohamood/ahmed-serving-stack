import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from threading import Thread
import pynvml

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

_util_samples = []
_sampler_thread = None
_sampler_running = False

def _sampler_worker():
    global _util_samples, _sampler_running
    while _sampler_running:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
            _util_samples.append(util)
            time.sleep(0.01)
        except:
            pass

def start_sampler():
    global _util_samples, _sampler_thread, _sampler_running
    _util_samples = []
    _sampler_running = True
    _sampler_thread = Thread(target=_sampler_worker, daemon=True)
    _sampler_thread.start()

def stop_sampler():
    global _sampler_running
    _sampler_running = False
    if _sampler_thread:
        _sampler_thread.join(timeout=1)

def read_util_mean() -> float:
    return sum(_util_samples) / len(_util_samples) if _util_samples else 0.0

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

def load(dtype: str):
    if dtype == "fp16":
        return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    if dtype == "int8":
        qc = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=qc, device_map="cuda")
    raise ValueError(dtype)

def make_prompt(context_tokens: int) -> str:
    base = "Summarise the following text in one sentence.\n"
    filler = ("The data center runs many small inference requests all day. " * 400)
    ids = tok(base + filler)["input_ids"][:context_tokens]
    return tok.decode(ids)

def resident_vram_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / (1024 ** 3)

def profile(model, dtype: str, context: int, new_tokens: int = 128, batch: int = 1):
    prompt = make_prompt(context)
    prompts = [prompt] * batch
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    _ = model.generate(**enc, max_new_tokens=8, do_sample=False)
    vram = resident_vram_gb()
    start_sampler()
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False)
    dt = time.time() - t0
    stop_sampler()
    gen_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * batch
    return {
        "dtype": dtype,
        "context": context,
        "vram_gb": round(vram, 3),
        "util_mean": round(read_util_mean(), 1),
        "tokens_per_s": round(gen_tokens / dt, 1),
    }

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
