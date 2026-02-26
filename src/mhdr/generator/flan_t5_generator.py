from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class FlanGenerator:
    def __init__(self, model_name="google/flan-t5-large"):  
        self.device = "cpu"  

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
    self,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        results = [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        return results