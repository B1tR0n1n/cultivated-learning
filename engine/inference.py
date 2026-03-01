import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from sentence_transformers import SentenceTransformer


class LogitBiasProcessor(LogitsProcessor):
    """Applies additive biases to specific token logits at each generation step.

    Used to suppress instruct-isms and user-corrected phrases. A bias of -10.0
    effectively removes a token from consideration without hard-blocking it.
    """

    def __init__(self, bias_map):
        self.bias_map = bias_map  # {token_id (int): bias (float)}

    def __call__(self, input_ids, scores):
        for token_id, bias in self.bias_map.items():
            scores[:, token_id] += bias
        return scores


class InferenceEngine:
    """Wrapper around the base LLM. All model interaction goes through here.
    
    v2 changes:
    - Default max_context raised from 4096 → 16384
    """
    
    def __init__(self, model_path, max_context=16384, 
                 embedding_model_path="/workspace/models/results/all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.max_context = max_context
        self.embedding_model_path = embedding_model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_model = None
        self._logit_biases = {}  # {token_id: bias} — set via set_logit_biases()
    
    def load(self):
        # Load Mistral for generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,   # FIX: was 'dtype' — HF silently ignored it, loading float32
            device_map="auto",
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.device
        
        # Load dedicated embedding model
        self.embedding_model = SentenceTransformer(
            self.embedding_model_path, device=str(self.device)
        )
        
        vram = torch.cuda.memory_allocated(0) / 1e9
        print(f"Loaded {self.model_path}")
        print(f"Loaded embedding model: {self.embedding_model_path}")
        print(f"  VRAM: {vram:.2f} GB")
        print(f"  Embedding dim: {self.embedding_model.get_sentence_embedding_dimension()}")
        print(f"  Max context: {self.max_context} tokens")
        return self
    
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context - max_new_tokens
        ).to(self.device)
        
        processors = LogitsProcessorList()
        if self._logit_biases:
            processors.append(LogitBiasProcessor(self._logit_biases))

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=processors,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        del inputs, output_ids
        torch.cuda.empty_cache()
        
        return response
    
    def generate_structured(self, prompt, max_new_tokens=512, temperature=0.3):
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.95)
    
    def get_embedding(self, text):
        """Generate embedding using dedicated sentence-transformer model (384-dim)."""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding
    
    def get_embedding_dimension(self):
        """Return the dimensionality of the embedding model."""
        return self.embedding_model.get_sentence_embedding_dimension()

    def generate_batch(self, prompts, max_new_tokens=512, temperature=0.3):
        """Generate responses for multiple prompts in a single forward pass.

        Uses left-padding so all sequences in the batch align at the right
        (where generation starts). Padding side is restored to 'right' after.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Max new tokens per response
            temperature: Sampling temperature

        Returns:
            List of response strings in the same order as input prompts
        """
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context - max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # All inputs are left-padded to the same length, so prompt_len is uniform
        prompt_len = inputs["input_ids"].shape[-1]
        responses = []
        for out in output_ids:
            new_tokens = out[prompt_len:]
            responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        del inputs, output_ids
        torch.cuda.empty_cache()

        self.tokenizer.padding_side = "right"
        return responses

    def set_logit_biases(self, bias_map):
        """Replace the active logit bias map. Called before each generation pass."""
        self._logit_biases = bias_map

    def suppress_tokens(self, phrases):
        """Tokenize phrases and build a bias map suppressing the first token of each.

        Applying -10.0 to the first token of a phrase is sufficient to prevent
        the model from opening with that phrase. Returns the bias map (does not
        set it — caller decides whether to apply it).

        Args:
            phrases: List of strings to suppress

        Returns:
            {token_id (int): -10.0} dict
        """
        bias_map = {}
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            if token_ids:
                bias_map[token_ids[0]] = -10.0
        return bias_map

    def cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two vectors.

        Since all-MiniLM-L6-v2 produces normalized embeddings (normalize_embeddings=True),
        the norms are always 1.0 and this reduces to a dot product. We keep the full
        formula for safety — if someone swaps to a non-normalizing model, this still works.

        Args:
            vec_a: First embedding vector (numpy array or list)
            vec_b: Second embedding vector (numpy array or list)

        Returns:
            Float between -1.0 and 1.0. Higher = more semantically similar.
        """
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
