import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


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
    
    def load(self):
        # Load Mistral for generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
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
