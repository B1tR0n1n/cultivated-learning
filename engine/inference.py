import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from sentence_transformers import SentenceTransformer


class LogitBiasProcessor(LogitsProcessor):
    """Applies additive biases to specific token logits at each generation step.

    Two suppression modes:

    flat_biases — {token_id: bias} applied unconditionally at every step.
    Used only for single-token phrases (e.g. "Certainly!" is one token).

    sequence_suppressions — list of token-ID sequences. For each sequence
    [t0, t1, ..., tN], tN is penalized only when input_ids currently ends
    with [t0, ..., t(N-1)]. This blocks a multi-word phrase from completing
    without broadly suppressing common component tokens like ▁I, ▁Of, ▁As.
    """

    def __init__(self, flat_biases, sequence_suppressions):
        self.flat_biases = flat_biases              # {token_id: float}
        self.sequence_suppressions = sequence_suppressions  # [[token_id, ...], ...]

    def __call__(self, input_ids, scores):
        for token_id, bias in self.flat_biases.items():
            scores[:, token_id] += bias

        if self.sequence_suppressions:
            tail = input_ids[0].tolist()
            for seq in self.sequence_suppressions:
                if len(seq) < 2:
                    continue
                prefix = seq[:-1]
                target_token = seq[-1]
                if tail[-len(prefix):] == prefix:
                    scores[:, target_token] += -10.0

        return scores


class InferenceEngine:
    """Wrapper around the base LLM. All model interaction goes through here.

    v3 changes (24b fork):
    - Default model path updated to Mistral-Small-24B-Instruct-2501-AWQ
    - Default max_context raised to 32768 for 24B context window
    - prompt_format property: auto-detected from num_hidden_layers after load()
      - 40 layers → "mistral-small" (Mistral Small 24B format)
      - other    → "mistral-v03"    (Mistral 7B v0.3 format)
    """

    def __init__(self, model_path="/workspace/models/results/Mistral-Small-24B-Instruct-2501-AWQ",
                 max_context=4096,
                 embedding_model_path="/workspace/models/results/all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.max_context = max_context
        self.embedding_model_path = embedding_model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_model = None
        self._flat_biases = {}             # {token_id: bias} for single-token suppressions
        self._sequence_suppressions = []   # [[token_id, ...]] for multi-token phrases
        self._prompt_format = "mistral-v03"  # overwritten in load()

    @property
    def prompt_format(self):
        return self._prompt_format

    def load(self):
        # Load Mistral for generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,          # Transformers 4.57+: use dtype, not torch_dtype
            device_map="auto",
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.device

        # Detect prompt format from model architecture
        layers = self.model.config.num_hidden_layers
        self._prompt_format = "mistral-small" if layers == 40 else "mistral-v03"

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
        print(f"  Prompt format: {self._prompt_format} ({layers} layers)")
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
        if self._flat_biases or self._sequence_suppressions:
            processors.append(LogitBiasProcessor(self._flat_biases, self._sequence_suppressions))

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

    def set_logit_biases(self, bias_data):
        """Store logit bias data returned by suppress_tokens().

        Args:
            bias_data: (flat_biases, sequence_suppressions) tuple
        """
        self._flat_biases, self._sequence_suppressions = bias_data

    def suppress_tokens(self, phrases):
        """Tokenize phrases and build suppression structures.

        Single-token phrases are added to a flat bias map and suppressed
        unconditionally at every generation step.

        Multi-token phrases are stored as sequences. Only the final token of
        each sequence is penalized, and only when input_ids currently ends with
        the preceding tokens. This blocks "Of course!" without suppressing the
        token ▁Of in unrelated contexts.

        Args:
            phrases: List of strings to suppress

        Returns:
            (flat_biases, sequence_suppressions) where:
              flat_biases:          {token_id (int): -10.0}
              sequence_suppressions: [[token_id (int), ...], ...]
        """
        flat_biases = {}
        sequence_suppressions = []
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            if not token_ids:
                continue
            if len(token_ids) == 1:
                flat_biases[token_ids[0]] = -10.0
            else:
                sequence_suppressions.append(token_ids)
        return flat_biases, sequence_suppressions

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
