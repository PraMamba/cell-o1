import os
import json
import tqdm
import argparse
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from typing import Optional
sys.path.append(".")
sys.path.append("./cell-o1")

cached_llms = dict()
cached_retrievers = dict()
cached_corpora = dict()

class LLMEngine:
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
        cache_dir: Optional[str] = None, 
        api: bool = False, 
        lora: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        shared_checkpoint: bool = True,
    ):
        self.llm_name = llm_name
        self.cache_dir = cache_dir or os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        self.api = api
        self.model_dtype = model_dtype
        global cached_llms
        new_llm = True
        if self.llm_name in cached_llms and shared_checkpoint:
            self.model = cached_llms[self.llm_name]
            new_llm = False
        if new_llm:
            model = AutoModelForCausalLM.from_pretrained(self.llm_name, device_map="auto", cache_dir=self.cache_dir, torch_dtype=self.model_dtype)
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if os.path.exists(self.llm_name) and os.path.exists(os.path.join(self.llm_name, "adapter_config.json")):
                lora = True
            if cache_dir and os.path.exists(os.path.join(cache_dir, "--".join(["models"] + self.llm_name.split('/')))) and os.path.exists(os.path.join(cache_dir, "--".join(["models"] + self.llm_name.split('/')), "adapter_config.json")):
                lora = True
            if lora:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.llm_name)
                model = model.merge_and_unload()
            else:
                pass
            self.model = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.model.model.config._name_or_path = self.llm_name
            self.model.model.generation_config.temperature=None
            self.model.model.generation_config.top_p=None
            if shared_checkpoint:
                cached_llms[self.llm_name] = self.model
        self.client = None

    def generate(
        self, 
        messages: list[dict[str, str]], 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_return_sequences: int = 1,
        **kwargs
    ):
        if self.model is not None:
            if temperature > 0.0:
                response = self.model(messages, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=temperature, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[i]["generated_text"][-1]["content"] for i in range(num_return_sequences)]
            else:
                response = self.model(messages, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[0]["generated_text"][-1]["content"]]
        else:
            response = self.client.chat.completions.create(
                model= self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=num_return_sequences if temperature > 0.0 else 1,
                **kwargs
            )
            outputs = [response.choices[i].message.content for i in range(num_return_sequences if temperature > 0.0 else 1)]
        return outputs


parser = argparse.ArgumentParser()
parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--i", type=int, default=0)
parser.add_argument("--folder", type=str, default="prediction")
parser.add_argument("--dataset", type=str, default="test.json")

args = parser.parse_args()

llm = LLMEngine(
    llm_name = args.llm_name,
    cache_dir = "../huggingface/hub",
)

save_dir = f"/mnt/data/{args.folder}/{args.llm_name.replace('/', '_')}"

dataset = json.load(open(args.dataset))
os.makedirs(save_dir, exist_ok=True)

curr_range = [j for j in range(len(dataset)) if j % args.n == args.i]

for i in tqdm.tqdm(curr_range):
    json_path = os.path.join(save_dir, f"{i}.json")
    regenerate = False
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
                pred = existing_data.get("prediction", "")
                if "OutOfMemoryError" in pred or "CUDA out of memory" in pred:
                    print(f"Prediction error in {json_path}, regenerating...")
                    regenerate = True
        except Exception:
            print(f"Invalid JSON at {json_path}, will regenerate.")
            regenerate = True

    else:
        regenerate = True
    
    if regenerate:
        item = dataset[i]
        try:
            pred = llm.generate(
                [
                    {
                        "role": "user",
                        "content": item["user_msg"] + '\n' + item["system_msg"]
                    }
                ],
                max_new_tokens = 3500
            )[0]
        except Exception as E:
            error_class = E.__class__.__name__
            pred = f"{error_class}: {E}"
        item["prediction"] = pred
        json.dump(item, open(os.path.join(save_dir, f"{i}.json"), 'w'), indent=4)
