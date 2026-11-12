"""training.policy: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from shopping_agent.harness.protocol import render_policy_prompt
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch


class TrainPolicy:
    def __init__(self, model_name: str, max_input_tokens: int, max_new_tokens: int,
                 init_adapter: str | None = None, sampling_temperature: float = 0.8,
                 json_grammar: bool = False, native_tools: bool = False,
                 governed_bundle_price: bool = False,
                 finish_quoted_ids: bool = False,
                 quote_decision_gate: bool = False):
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        if not 0 < sampling_temperature <= 2:
            raise ValueError("sampling_temperature must be in (0, 2]")
        self.sampling_temperature = sampling_temperature
        if json_grammar and native_tools:
            raise ValueError("JSON grammar and native tool calls cannot be combined")
        self.native_tools = native_tools
        self.governed_bundle_price = governed_bundle_price
        self.finish_quoted_ids = finish_quoted_ids
        self.quote_decision_gate = quote_decision_gate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.json_grammar = None
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.model = (PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
                      if init_adapter else get_peft_model(base, LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )))
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.enable_input_require_grads()
        if json_grammar:
            import xgrammar as xgr
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=self.model.config.vocab_size)
            self.json_grammar = xgr.GrammarCompiler(tokenizer_info).compile_builtin_json_grammar()

    def generate(self, messages: list[dict[str, str]], seed: int) -> tuple[str, list[int], list[int]]:
        text = render_policy_prompt(self.tokenizer, messages, self.native_tools,
                                    self.governed_bundle_price,
                                    self.finish_quoted_ids, self.quote_decision_gate)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        ).to("cuda")
        if encoded.input_ids.shape[1] > self.max_input_tokens:
            raise RuntimeError(
                f"Input has {encoded.input_ids.shape[1]} tokens, above the configured "
                f"{self.max_input_tokens}; refusing silent truncation of history or tool state"
            )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.eval()
        with torch.no_grad():
            grammar_kwargs = {}
            if self.json_grammar is not None:
                from xgrammar.contrib.hf import LogitsProcessor
                grammar_kwargs["logits_processor"] = [LogitsProcessor(self.json_grammar)]
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.sampling_temperature,
                top_p=0.95,
                top_k=0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **grammar_kwargs,
            )
        prompt_ids = encoded.input_ids[0].detach().cpu().tolist()
        response_ids = output[0, encoded.input_ids.shape[1]:].detach().cpu().tolist()
        raw = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return raw, prompt_ids, response_ids

    def generate_many(self, requests: list[tuple[list[dict[str, str]], int]]):
        """One padded forward pass for active trajectories at the same turn."""
        if len(requests) == 1:
            messages, seed = requests[0]
            return [self.generate(messages, seed)]
        texts = [render_policy_prompt(self.tokenizer, messages, self.native_tools,
                                      self.governed_bundle_price,
                                      self.finish_quoted_ids, self.quote_decision_gate)
            for messages, _ in requests]
        ids = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
        longest = max(map(len, ids))
        if longest > self.max_input_tokens:
            raise RuntimeError(f"Batch prompt has {longest} tokens, above {self.max_input_tokens}")
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer.pad(
                {"input_ids": ids, "attention_mask": [[1] * len(value) for value in ids]},
                padding=True, return_tensors="pt",
            ).to("cuda")
        finally:
            self.tokenizer.padding_side = original_padding_side
        # The batch seed is stable for a given request set. Individual draws
        # can differ from serial generation, so before/after use the same mode.
        torch.manual_seed(requests[0][1])
        torch.cuda.manual_seed_all(requests[0][1])
        self.model.eval()
        with torch.no_grad():
            grammar_kwargs = {}
            if self.json_grammar is not None:
                from xgrammar.contrib.hf import LogitsProcessor
                grammar_kwargs["logits_processor"] = [LogitsProcessor(self.json_grammar)]
            output = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=self.sampling_temperature, top_p=0.95, top_k=0,
                pad_token_id=self.tokenizer.eos_token_id, use_cache=True,
                **grammar_kwargs,
            )
        result = []
        for index, prompt_ids in enumerate(ids):
            response_ids = output[index, longest:].detach().cpu().tolist()
            # generate() pads a finished row until the slowest row finishes.
            # Padding after its first EOS is not a model action and must not
            # enter the trajectory, reward, or policy loss.
            if self.tokenizer.eos_token_id in response_ids:
                response_ids = response_ids[:response_ids.index(self.tokenizer.eos_token_id) + 1]
            raw = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            result.append((raw, prompt_ids, response_ids))
        return result
