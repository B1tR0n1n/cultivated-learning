# Cultivated Learning — Session 3 Patch Notes

**Date:** February 28, 2026  
**Prepared by:** Claude (Opus 4.6) via claude.ai project  
**For:** Claude Code in VS Code / container terminal  
**Project path:** `/workspace/Projects/cultivated-learning/`

---

## Summary

5 files modified. 3 bugs fixed, 1 code quality improvement (cosine dedup), 2 enhancements (reload directives, env vars). All fixes are backward-compatible with existing memory databases — no data migration needed.

**IMPORTANT:** Before applying any changes, back up the current files:

```bash
cd /workspace/Projects/cultivated-learning
mkdir -p backups/session3
cp engine/inference.py backups/session3/
cp core/memory_store.py backups/session3/
cp core/reflection.py backups/session3/
cp core/cold_storage.py backups/session3/
cp ui/app.py backups/session3/
```

---

## Fix 1 — `inference.py` (Bug: VRAM doubled)

**File:** `engine/inference.py`  
**Severity:** Bug — silent, wastes ~7GB VRAM  

### What's wrong

Line 26 uses `dtype=torch.float16`. The correct Hugging Face parameter is `torch_dtype`. HF silently ignores unknown kwargs, so the model loads in float32 (~14GB) instead of float16 (~7GB). The 5090's 34GB masks the problem, but VRAM headroom numbers in all session logs have been inaccurate.

### Changes

1. **Line 26:** `dtype=torch.float16` → `torch_dtype=torch.float16`
2. **Added:** `import numpy as np` at top
3. **Added:** `cosine_similarity(self, vec_a, vec_b)` method at bottom — shared utility replacing 3 separate manual implementations across cold_storage.py, reflection.py, and memory_store.py. Includes zero-norm guard and works with both normalized and non-normalized embeddings.

### How to verify

After applying, reload the model and check VRAM:
```python
import torch
print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
# Should be ~7.5GB, not ~14.5GB
```

### Full replacement file

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


class InferenceEngine:
    """Wrapper around the base LLM. All model interaction goes through here."""
    
    def __init__(self, model_path, max_context=4096, 
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
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
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
```

---

## Fix 2 — `memory_store.py` (Bug: superseded memory leak)

**File:** `core/memory_store.py`  
**Severity:** Bug — architectural, undermines the correction system  

### What's wrong

`retrieve_by_type()` returns superseded memories. The consolidation engine calls `retrieve_by_type(MemoryType.EPISODIC)` to find fading memories to distill. If those fading episodes were previously superseded by a correction, they get consolidated into new semantic facts — resurrecting wrong information through a back door. This directly undermines the supersession mechanism built to break hallucination loops.

### Changes

1. **`retrieve_by_type()`:** Added `if mem.superseded_by is None` filter, matching `retrieve()` behavior
2. **Added:** `supersede(old_memory_id, new_memory_id)` method — clean API for marking supersession instead of reaching into collection metadata directly

### Full replacement file

```python
import json
import time
import math
import uuid
import chromadb
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from enum import Enum


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"


@dataclass
class MemoryUnit:
    content: str
    memory_type: MemoryType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_interaction_id: str = ""
    salience_score: float = 0.5
    confidence: float = 0.7
    superseded_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_metadata(self):
        return {
            "memory_type": self.memory_type.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source_interaction_id": self.source_interaction_id,
            "salience_score": self.salience_score,
            "confidence": self.confidence,
            "superseded_by": self.superseded_by or "",
            "tags": json.dumps(self.tags),
        }

    @staticmethod
    def from_chroma(id, document, metadata, embedding=None):
        return MemoryUnit(
            id=id,
            content=document,
            memory_type=MemoryType(metadata["memory_type"]),
            created_at=metadata["created_at"],
            last_accessed=metadata["last_accessed"],
            access_count=metadata["access_count"],
            source_interaction_id=metadata.get("source_interaction_id", ""),
            salience_score=metadata["salience_score"],
            confidence=metadata["confidence"],
            superseded_by=metadata.get("superseded_by", "") or None,
            tags=json.loads(metadata.get("tags", "[]")),
        )


class MemoryStore:
    """Persistent semantic memory with vector search and salience decay."""

    def __init__(self, persist_dir, engine=None, decay_rate=0.01):
        self.engine = engine
        self.decay_rate = decay_rate
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="cultivated_memory",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Memory store initialized: {self.collection.count()} existing memories")

    def store(self, memory: MemoryUnit, embedding=None):
        if embedding is None and self.engine is not None:
            embedding = self.engine.get_embedding(memory.content)

        self.collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[memory.to_metadata()],
            embeddings=[embedding.tolist()] if embedding is not None else None,
        )
        return memory.id

    def retrieve(self, query_text, top_k=10, min_salience=0.1,
                 similarity_weight=0.6, salience_weight=0.4):
        if self.collection.count() == 0:
            return []

        if self.engine is not None:
            query_embedding = self.engine.get_embedding(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 3, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k * 3, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        scored_memories = []
        for i in range(len(results["ids"][0])):
            mem = MemoryUnit.from_chroma(
                id=results["ids"][0][i],
                document=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            if mem.salience_score >= min_salience and mem.superseded_by is None:
                mem.last_accessed = time.time()
                mem.access_count += 1
                self._update_access(mem)
                similarity = 1 - results["distances"][0][i]  # cosine distance → similarity
                blended_score = (similarity * similarity_weight) + (mem.salience_score * salience_weight)
                scored_memories.append((mem, blended_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored_memories[:top_k]]

    def retrieve_by_type(self, memory_type: MemoryType, limit=20):
        """Retrieve memories of a specific type, excluding superseded ones.
        
        FIX: Previously returned superseded memories, which meant the consolidation
        engine could pull in corrected episodic memories and distill them into new
        semantic facts — resurrecting wrong information through a back door.
        Now filters superseded memories the same way retrieve() does.
        """
        results = self.collection.get(
            where={"memory_type": memory_type.value},
            limit=limit,
        )
        memories = []
        for i in range(len(results["ids"])):
            mem = MemoryUnit.from_chroma(
                id=results["ids"][i],
                document=results["documents"][i],
                metadata=results["metadatas"][i],
            )
            # FIX: filter out superseded memories (matches retrieve() behavior)
            if mem.superseded_by is None:
                memories.append(mem)
        return memories

    def adjust_salience(self, memory_id, delta):
        results = self.collection.get(ids=[memory_id])
        if not results["ids"]:
            return
        metadata = results["metadatas"][0]
        metadata["salience_score"] = max(0.0, min(1.0, metadata["salience_score"] + delta))
        self.collection.update(ids=[memory_id], metadatas=[metadata])

    def supersede(self, old_memory_id, new_memory_id):
        """Mark a memory as superseded by another. Superseded memories are
        excluded from all retrieval paths (retrieve, retrieve_by_type).
        
        Args:
            old_memory_id: The memory being replaced
            new_memory_id: The memory that replaces it
        """
        results = self.collection.get(ids=[old_memory_id])
        if not results["ids"]:
            return
        metadata = results["metadatas"][0]
        metadata["superseded_by"] = new_memory_id
        self.collection.update(ids=[old_memory_id], metadatas=[metadata])

    def decay_pass(self):
        all_memories = self.collection.get()
        updated_ids = []
        updated_metadatas = []
        for i in range(len(all_memories["ids"])):
            metadata = all_memories["metadatas"][i]
            hours_since_access = (time.time() - metadata["last_accessed"]) / 3600
            decay_factor = math.exp(-self.decay_rate * hours_since_access)
            new_salience = metadata["salience_score"] * decay_factor
            if abs(new_salience - metadata["salience_score"]) > 0.001:
                metadata["salience_score"] = new_salience
                updated_ids.append(all_memories["ids"][i])
                updated_metadatas.append(metadata)

        if updated_ids:
            self.collection.update(ids=updated_ids, metadatas=updated_metadatas)
        print(f"Decay pass: updated {len(updated_ids)} of {len(all_memories['ids'])} memories")

    def get_stats(self):
        all_memories = self.collection.get()
        total = len(all_memories["ids"])
        if total == 0:
            return {"total": 0}

        types = {}
        saliences = []
        for m in all_memories["metadatas"]:
            t = m["memory_type"]
            types[t] = types.get(t, 0) + 1
            saliences.append(m["salience_score"])

        return {
            "total": total,
            "by_type": types,
            "avg_salience": sum(saliences) / len(saliences),
            "min_salience": min(saliences),
            "max_salience": max(saliences),
        }

    def _update_access(self, memory: MemoryUnit):
        self.collection.update(
            ids=[memory.id],
            metadatas=[memory.to_metadata()],
        )
```

---

## Fix 3 — `cold_storage.py` (Cosine dedup)

**File:** `core/cold_storage.py`  
**Severity:** Code quality  

### What's wrong

Manual cosine similarity computation with numpy duplicated across 3 files. If the embedding model changes (e.g., from normalized to non-normalized), you'd have to find and fix all three.

### Changes

1. **Removed:** `import numpy as np`
2. **`resurface()`:** Replaced manual `np.dot / (norm * norm)` with `self.engine.cosine_similarity()`

### Full replacement file

```python
import time
import json
import os
from core.memory_store import MemoryUnit, MemoryType


class ColdStorage:
    """Archive for memories that fade below the salience floor.
    Resurfaces archived memories when an unusually strong semantic match occurs."""

    def __init__(self, engine, memory_store, archive_dir, salience_floor=0.15, resurface_threshold=0.75):
        self.engine = engine
        self.memory = memory_store
        self.archive_dir = archive_dir
        self.salience_floor = salience_floor
        self.resurface_threshold = resurface_threshold
        os.makedirs(archive_dir, exist_ok=True)
        self.archive = self._load_archive()
        print(f"Cold storage initialized: {len(self.archive)} archived memories")

    def _load_archive(self):
        archive_path = os.path.join(self.archive_dir, "cold_archive.json")
        if os.path.exists(archive_path):
            with open(archive_path, "r") as f:
                return json.load(f)
        return []

    def _save_archive(self):
        archive_path = os.path.join(self.archive_dir, "cold_archive.json")
        with open(archive_path, "w") as f:
            json.dump(self.archive, f, indent=2)

    def archive_pass(self):
        all_data = self.memory.collection.get(include=["documents", "metadatas", "embeddings"])
        archived_count = 0
        ids_to_delete = []

        for i in range(len(all_data["ids"])):
            metadata = all_data["metadatas"][i]
            if metadata["salience_score"] < self.salience_floor:
                embedding = None
                if all_data["embeddings"] is not None:
                    embedding = [float(x) for x in all_data["embeddings"][i]]

                entry = {
                    "id": all_data["ids"][i],
                    "document": all_data["documents"][i],
                    "metadata": metadata,
                    "embedding": embedding,
                    "archived_at": time.time(),
                }
                self.archive.append(entry)
                ids_to_delete.append(all_data["ids"][i])
                archived_count += 1

        # Save archive BEFORE deleting from active memory
        if archived_count > 0:
            self._save_archive()
            self.memory.collection.delete(ids=ids_to_delete)

        print(f"Archive pass: {archived_count} memories moved to cold storage. "
              f"Total archived: {len(self.archive)}")
        return archived_count

    def resurface(self, query_text):
        if not self.archive:
            return []

        query_embedding = self.engine.get_embedding(query_text)
        resurfaced = []

        for entry in self.archive:
            if entry["embedding"] is None:
                continue

            # FIX: Use engine's shared cosine_similarity instead of manual numpy
            similarity = self.engine.cosine_similarity(query_embedding, entry["embedding"])

            if similarity >= self.resurface_threshold:
                mem = MemoryUnit.from_chroma(
                    id=entry["id"],
                    document=entry["document"],
                    metadata=entry["metadata"],
                )
                mem.salience_score = 0.5
                mem.last_accessed = time.time()
                mem.tags.append("resurfaced")
                self.memory.store(mem)
                resurfaced.append(mem)

        if resurfaced:
            resurfaced_ids = {m.id for m in resurfaced}
            self.archive = [e for e in self.archive if e["id"] not in resurfaced_ids]
            self._save_archive()
            print(f"Resurfaced {len(resurfaced)} memories from cold storage!")

        return resurfaced

    def get_stats(self):
        return {
            "archived_count": len(self.archive),
            "salience_floor": self.salience_floor,
            "resurface_threshold": self.resurface_threshold,
        }
```

---

## Fix 4 — `reflection.py` (Cosine dedup + dedup gate)

**File:** `core/reflection.py`  
**Severity:** Code quality + enhancement  

### Changes

1. **`_depth_2()`:** Replaced manual cosine with `self.engine.cosine_similarity()`
2. **`_depth_2()`:** Added semantic deduplication gate — new directives are compared against all existing ones at threshold 0.60 before being accepted. This was documented in the Session 2 handoff as needed but wasn't in the code yet.
3. **`_load_directives()`:** Docstring clarified that this method is also callable from the UI for reload-after-prune scenarios.

### Full replacement file

```python
import time
from core.memory_store import MemoryUnit, MemoryType


class ReflectionEngine:
    """Post-interaction recursive self-analysis at increasing depths."""

    def __init__(self, engine, memory_store, max_depth=3):
        self.engine = engine
        self.memory = memory_store
        self.max_depth = max_depth
        self.directives = []
        self._load_directives()

    def _load_directives(self):
        """Load existing procedural directives from memory.
        
        Also callable from the UI to force a reload after manual pruning.
        Without this, manually deleting directives from ChromaDB leaves the
        in-memory list stale until the next full restart.
        """
        procedural = self.memory.retrieve_by_type(MemoryType.PROCEDURAL)
        self.directives = [m.content for m in procedural]
        print(f"Reflection engine loaded {len(self.directives)} existing directives.")

    def reflect(self, user_message, assistant_response, interaction_id):
        """Run reflection pass at all depths. Returns list of new memories created."""
        new_memories = []

        # Depth 0 — Factual: What happened?
        d0 = self._depth_0(user_message, assistant_response)
        if d0:
            new_memories.append(d0)

        # Depth 1 — Analytical: What patterns emerge?
        d1 = self._depth_1(user_message, assistant_response)
        if d1:
            new_memories.append(d1)

        # Depth 2 — Prescriptive: What should change?
        d2 = self._depth_2(d0, d1)
        if d2:
            new_memories.append(d2)

        # Depth 3 — Meta-coherence: Are directives consistent?
        if d2:
            d3 = self._depth_3()
            if d3:
                new_memories.append(d3)

        # Store all new memories
        for mem in new_memories:
            mem.source_interaction_id = interaction_id
            self.memory.store(mem)

        return new_memories

    def _depth_0(self, user_message, assistant_response):
        """Factual: evaluate what happened in this interaction."""
        prompt = (
            "[INST] You are a self-reflection module analyzing an interaction.\n\n"
            f"Interaction:\nUser: {user_message}\nAssistant: {assistant_response}\n\n"
            "Analyze this interaction factually:\n"
            "1. Was the response accurate and relevant?\n"
            "2. Did it address what the user actually asked?\n"
            "3. Were there any errors or misunderstandings?\n\n"
            "Be brief and specific. One paragraph. [/INST]"
        )

        analysis = self.engine.generate_structured(prompt, max_new_tokens=200)

        return MemoryUnit(
            content=f"Reflection D0: {analysis}",
            memory_type=MemoryType.REFLECTIVE,
            salience_score=0.4,
            confidence=0.6,
            tags=["reflection", "depth_0", "factual"],
        )

    def _depth_1(self, user_message, assistant_response):
        """Analytical: identify patterns across recent interactions."""
        recent = self.memory.retrieve(user_message, top_k=5)
        if len(recent) < 2:
            return None

        memory_context = "\n".join(
            [f"- [{m.memory_type.value}] {m.content[:150]}" for m in recent]
        )

        prompt = (
            "[INST] You are a self-reflection module analyzing patterns.\n\n"
            f"Current interaction:\nUser: {user_message}\nAssistant: {assistant_response}\n\n"
            f"Recent memories:\n{memory_context}\n\n"
            "What patterns do you notice?\n"
            "- Recurring user needs or preferences\n"
            "- Consistent strengths or weaknesses in responses\n"
            "- Emerging themes across interactions\n\n"
            "Be brief and specific. One paragraph. [/INST]"
        )

        analysis = self.engine.generate_structured(prompt, max_new_tokens=200)

        return MemoryUnit(
            content=f"Reflection D1: {analysis}",
            memory_type=MemoryType.REFLECTIVE,
            salience_score=0.5,
            confidence=0.5,
            tags=["reflection", "depth_1", "analytical"],
        )

    def _depth_2(self, d0_memory, d1_memory):
        """Prescriptive: generate behavioral directives from analysis."""
        if not d0_memory and not d1_memory:
            return None

        context_parts = []
        if d0_memory:
            context_parts.append(f"Factual analysis: {d0_memory.content}")
        if d1_memory:
            context_parts.append(f"Pattern analysis: {d1_memory.content}")

        current_directives = "\n".join(
            [f"- {d}" for d in self.directives]
        ) if self.directives else "None yet."

        prompt = (
            "[INST] You are a self-reflection module generating behavioral directives.\n\n"
            f"Analysis:\n" + "\n".join(context_parts) + "\n\n"
            f"Current directives:\n{current_directives}\n\n"
            "Based on the analysis, should any NEW directive be added? A directive is a specific behavioral rule like:\n"
            '- "Keep responses under 3 sentences unless asked for detail"\n'
            '- "When user mentions Contact Front, ask about progress"\n\n'
            "Rules:\n"
            "- Only propose a directive if the analysis clearly supports it\n"
            "- Do not duplicate existing directives\n"
            "- If no new directive is needed, respond with exactly: NO_NEW_DIRECTIVE\n\n"
            "If proposing a directive, state it as a single clear sentence. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=100)

        if "NO_NEW_DIRECTIVE" in result.upper():
            return None

        # Clean up the directive
        directive = result.strip().split("\n")[0].strip()
        if len(directive) < 10 or len(directive) > 200:
            return None

        # Deduplication check: compare against existing directives using cosine similarity
        # Threshold 0.60 catches paraphrased versions that the 7B model generates
        directive_embedding = self.engine.get_embedding(directive)
        for existing in self.directives:
            existing_embedding = self.engine.get_embedding(existing)
            # FIX: Use engine's shared cosine_similarity instead of manual numpy
            similarity = self.engine.cosine_similarity(directive_embedding, existing_embedding)
            if similarity > 0.60:
                print(f"  Directive rejected (duplicate, sim={similarity:.2f}): {directive[:60]}")
                return None

        self.directives.append(directive)

        return MemoryUnit(
            content=directive,
            memory_type=MemoryType.PROCEDURAL,
            salience_score=0.8,
            confidence=0.7,
            tags=["reflection", "depth_2", "directive", "auto_generated"],
        )

    def _depth_3(self):
        """Meta-coherence: check directives for contradictions."""
        if len(self.directives) < 2:
            return None

        directive_list = "\n".join(
            [f"{i+1}. {d}" for i, d in enumerate(self.directives)]
        )

        prompt = (
            "[INST] You are a coherence checker for behavioral directives.\n\n"
            f"Current directives:\n{directive_list}\n\n"
            "Check for:\n"
            "1. Contradictions between directives\n"
            "2. Redundancies (two directives saying the same thing)\n"
            "3. Directives that are too vague to be actionable\n\n"
            "If all directives are coherent, respond with exactly: COHERENT\n\n"
            "If there are issues, briefly describe each one and which directive numbers are involved. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=200)

        if "COHERENT" in result.upper():
            return None

        return MemoryUnit(
            content=f"Reflection D3 — Coherence issue: {result}",
            memory_type=MemoryType.REFLECTIVE,
            salience_score=0.6,
            confidence=0.5,
            tags=["reflection", "depth_3", "coherence"],
        )

    def get_directives(self):
        """Return current active directives for context assembly."""
        return self.directives.copy()
```

---

## Fix 5 — `app.py` (Bug: history desync + env vars + reload button)

**File:** `ui/app.py`  
**Severity:** Bug + enhancement  

### What's wrong

The `chat()` function maintained its own history list separate from `loop.history`. These two lists drifted independently — Gradio showed one thing, the model saw another. If Gradio reset or the user refreshed, they'd desync silently.

### Changes

1. **`chat()`:** Now builds Gradio history from `loop.history` as single source of truth. When loop trims to 20 messages, the UI reflects it — intentional, so the user sees what the model sees.
2. **Paths:** All hardcoded `/workspace/...` paths replaced with `os.environ.get()` with sensible defaults. Set `CL_BASE_DIR` and `CL_MODEL_PATH` env vars to override.
3. **`browse_memories()`:** "All" view now also filters superseded memories.
4. **`reload_directives()`:** New function + UI button under the Directives section. Fixes the stale directive problem after manual ChromaDB pruning.
5. **Layout:** Added "Reload from Memory" button next to "Show Active Directives".

### Full replacement file

```python
import sys
import os

# FIX: Use environment variable with sensible default instead of hardcoded path
BASE_DIR = os.environ.get("CL_BASE_DIR", "/workspace/Projects/cultivated-learning")
MODEL_PATH = os.environ.get("CL_MODEL_PATH", "/workspace/models/results/Mistral-7B-Instruct-v0.3")

sys.path.insert(0, BASE_DIR)

import gradio as gr
from engine.inference import InferenceEngine
from core.memory_store import MemoryStore, MemoryType
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop
from core.consolidation import ConsolidationEngine
from core.cold_storage import ColdStorage

# Initialize
print("Loading models...")
engine = InferenceEngine(MODEL_PATH)
engine.load()

memory = MemoryStore(
    persist_dir=os.path.join(BASE_DIR, "data/memory_db"),
    engine=engine
)

assembler = ContextAssembler(engine=engine, memory_store=memory)

loop = InteractionLoop(
    engine=engine,
    memory_store=memory,
    assembler=assembler,
    log_dir=os.path.join(BASE_DIR, "data/interaction_log")
)

consolidator = ConsolidationEngine(engine, memory)

cold = ColdStorage(
    engine=engine,
    memory_store=memory,
    archive_dir=os.path.join(BASE_DIR, "data/cold_storage")
)

print("All systems ready.")


# --- Functions ---

def chat(message, history):
    """Process a chat message through the full pipeline.
    
    FIX: Previously maintained a separate history list from loop.history,
    causing desync between what the user sees and what the model sees.
    Now uses loop.history as the single source of truth.
    
    Note: When loop.history trims to 20 messages (10 turns), the chatbox
    will also lose older messages. This is intentional — the UI should
    reflect what the model actually has access to. If the model can't see
    a message, showing it to the user creates false expectations.
    """
    response = loop.chat(message)
    # Build Gradio history from loop's authoritative history
    gradio_history = [
        {"role": h["role"], "content": h["content"]}
        for h in loop.history
    ]
    return "", gradio_history


def give_feedback(rating, correction):
    rating = int(rating)
    corr = correction.strip() if correction.strip() else None
    loop.feedback(rating=rating, correction=corr)
    msg = f"Rating {rating} recorded."
    if corr:
        msg += f" Correction stored at salience 0.9."
    return msg


def get_status():
    stats = memory.get_stats()
    status = loop.status()
    cold_stats = cold.get_stats()
    lines = [
        f"INTERACTION COUNT:  {status.get('interaction_count', '?')}",
        f"ACTIVE MEMORIES:    {stats['total']}",
        f"COLD STORAGE:       {cold_stats['archived_count']}",
        f"ACTIVE DIRECTIVES:  {status.get('active_directives', '?')}",
        f"REFLECTION:         {'ENABLED' if status.get('reflection_enabled') else 'DISABLED'}",
        "",
    ]
    if stats["total"] > 0:
        lines.append(f"BY TYPE:  {stats['by_type']}")
        lines.append(f"SALIENCE: avg {stats['avg_salience']:.3f}  min {stats['min_salience']:.3f}  max {stats['max_salience']:.3f}")
    return "\n".join(lines)


def run_consolidation():
    new_mems = consolidator.consolidate()
    if new_mems:
        results = [f"  [{m.salience_score:.2f}] {m.content[:100]}" for m in new_mems]
        return f"Consolidated {len(new_mems)} new semantic memories:\n" + "\n".join(results)
    return "No memories ready for consolidation."


def run_decay():
    memory.decay_pass()
    stats = memory.get_stats()
    return f"Decay pass complete. Avg salience: {stats['avg_salience']:.3f}"


def run_archive():
    count = cold.archive_pass()
    return f"Archive pass complete. {count} memories moved to cold storage."


def run_resurface(query):
    if not query.strip():
        return "Enter a query to check cold storage for matches."
    resurfaced = cold.resurface(query.strip())
    if resurfaced:
        results = [f"  [{m.salience_score:.2f}] {m.content[:100]}" for m in resurfaced]
        return f"Resurfaced {len(resurfaced)} memories:\n" + "\n".join(results)
    return "No matches in cold storage."


def browse_memories(mem_type):
    type_map = {
        "All": None,
        "Episodic": MemoryType.EPISODIC,
        "Semantic": MemoryType.SEMANTIC,
        "Procedural": MemoryType.PROCEDURAL,
        "Reflective": MemoryType.REFLECTIVE,
    }
    mt = type_map.get(mem_type)
    if mt:
        mems = memory.retrieve_by_type(mt, limit=30)
    else:
        all_data = memory.collection.get(limit=30)
        from core.memory_store import MemoryUnit
        mems = []
        for i in range(len(all_data["ids"])):
            mem = MemoryUnit.from_chroma(
                id=all_data["ids"][i],
                document=all_data["documents"][i],
                metadata=all_data["metadatas"][i],
            )
            # Also filter superseded in the "All" view
            if mem.superseded_by is None:
                mems.append(mem)

    mems.sort(key=lambda m: m.salience_score, reverse=True)
    lines = []
    for m in mems:
        lines.append(f"[{m.memory_type.value:11s}] sal:{m.salience_score:.2f}  {m.content[:90]}")
    return "\n".join(lines) if lines else "No memories found."


def get_directives():
    if loop.reflection_engine:
        directives = loop.reflection_engine.get_directives()
        if directives:
            return "\n".join([f"{i+1}. {d}" for i, d in enumerate(directives)])
    return "No active directives."


def reload_directives():
    """Reload directives from memory store into the reflection engine.
    
    Fixes the stale directive problem: if you manually prune directives
    through ChromaDB directly, the in-memory self.directives list goes
    stale until restart. This forces a reload without restarting.
    """
    if loop.reflection_engine:
        loop.reflection_engine._load_directives()
        count = len(loop.reflection_engine.get_directives())
        return f"Reloaded {count} directives from memory store."
    return "Reflection engine not enabled."


# --- Theme ---

b1tr0n1n = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fdf8e8", c100="#f5ebc4", c200="#edde9f", c300="#e4d07b",
        c400="#d4b83e", c500="#c9a227", c600="#a88520", c700="#8b7320",
        c800="#6e5a19", c900="#514213", c950="#3a2f0e"
    ),
    neutral_hue=gr.themes.Color(
        c50="#ede5d0", c100="#c8bda0", c200="#9a9080", c300="#7a7060",
        c400="#5a5245", c500="#3d362c", c600="#2a2620", c700="#1a1814",
        c800="#0f0e0b", c900="#0a0908", c950="#050504"
    ),
    font=gr.themes.GoogleFont("JetBrains Mono"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0908",
    body_text_color="#c8bda0",
    block_background_fill="#1a1814",
    block_border_color="#2a2620",
    block_label_text_color="#7a7060",
    block_title_text_color="#ede5d0",
    input_background_fill="#0f0e0b",
    input_border_color="#2a2620",
    input_placeholder_color="#5a5245",
    button_primary_background_fill="#c9a227",
    button_primary_text_color="#0a0908",
    button_primary_background_fill_hover="#e4d07b",
    button_secondary_background_fill="#1a1814",
    button_secondary_text_color="#c8bda0",
    button_secondary_border_color="#3d362c",
    button_secondary_background_fill_hover="#2a2620",
    checkbox_background_color="#0f0e0b",
    checkbox_border_color="#3d362c",
    slider_color="#c9a227",
)


# --- Layout ---

with gr.Blocks(title="Cultivated Learning", theme=b1tr0n1n, css="""
    .gradio-container { max-width: 900px !important; }
    footer { display: none !important; }
    .tab-nav button { font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
""") as app:

    gr.Markdown("# 🌱 CULTIVATED LEARNING")
    gr.Markdown("*Frozen model. Growing mind. — b1tr0n1n*")

    with gr.Tab("Chat"):
        chatbox = gr.Chatbot(label="Conversation", height=420)
        with gr.Row():
            msg = gr.Textbox(label="Message", placeholder="Speak...", scale=5)
            send_btn = gr.Button("Send", variant="primary", scale=1)
        send_btn.click(fn=chat, inputs=[msg, chatbox], outputs=[msg, chatbox])
        msg.submit(fn=chat, inputs=[msg, chatbox], outputs=[msg, chatbox])

    with gr.Tab("Feedback"):
        gr.Markdown("### Signal")
        rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating")
        correction = gr.Textbox(label="Correction", placeholder="Behavioral correction (optional)")
        fb_btn = gr.Button("Submit", variant="primary")
        fb_output = gr.Textbox(label="Result", interactive=False)
        fb_btn.click(fn=give_feedback, inputs=[rating, correction], outputs=fb_output)

    with gr.Tab("Memory"):
        gr.Markdown("### Browse")
        with gr.Row():
            mem_type = gr.Dropdown(
                choices=["All", "Episodic", "Semantic", "Procedural", "Reflective"],
                value="All", label="Filter by type"
            )
            browse_btn = gr.Button("Browse", variant="primary")
        mem_output = gr.Textbox(label="Memories", interactive=False, lines=15)
        browse_btn.click(fn=browse_memories, inputs=[mem_type], outputs=mem_output)

        gr.Markdown("---")
        gr.Markdown("### Directives")
        with gr.Row():
            dir_btn = gr.Button("Show Active Directives", variant="secondary")
            reload_btn = gr.Button("Reload from Memory", variant="secondary")
        dir_output = gr.Textbox(label="Directives", interactive=False, lines=6)
        dir_btn.click(fn=get_directives, outputs=dir_output)
        reload_btn.click(fn=reload_directives, outputs=dir_output)

    with gr.Tab("Maintenance"):
        gr.Markdown("### Consolidation")
        gr.Markdown("Distill fading episodic memories into durable semantic memories.")
        consol_btn = gr.Button("Run Consolidation", variant="primary")
        consol_output = gr.Textbox(label="Result", interactive=False, lines=6)
        consol_btn.click(fn=run_consolidation, outputs=consol_output)

        gr.Markdown("---")
        gr.Markdown("### Decay")
        gr.Markdown("Apply salience decay to all memories.")
        decay_btn = gr.Button("Run Decay Pass", variant="secondary")
        decay_output = gr.Textbox(label="Result", interactive=False)
        decay_btn.click(fn=run_decay, outputs=decay_output)

        gr.Markdown("---")
        gr.Markdown("### Cold Storage")
        gr.Markdown("Archive faded memories. Resurface on strong semantic match.")
        with gr.Row():
            archive_btn = gr.Button("Run Archive Pass", variant="secondary")
            archive_output = gr.Textbox(label="Result", interactive=False)
        archive_btn.click(fn=run_archive, outputs=archive_output)

        with gr.Row():
            resurface_query = gr.Textbox(label="Resurface Query", placeholder="Search cold storage...")
            resurface_btn = gr.Button("Resurface", variant="secondary")
        resurface_output = gr.Textbox(label="Result", interactive=False, lines=4)
        resurface_btn.click(fn=run_resurface, inputs=[resurface_query], outputs=resurface_output)

    with gr.Tab("Status"):
        status_output = gr.Textbox(label="System Status", interactive=False, lines=10)
        status_btn = gr.Button("Refresh", variant="primary")
        status_btn.click(fn=get_status, outputs=status_output)


app.launch(server_name="0.0.0.0", server_port=7880, share=False)
```

---

## Quick Apply (Container Shell)

If you want to apply all fixes at once from the container:

```bash
cd /workspace/Projects/cultivated-learning

# Backup first
mkdir -p backups/session3
cp engine/inference.py backups/session3/
cp core/memory_store.py backups/session3/
cp core/reflection.py backups/session3/
cp core/cold_storage.py backups/session3/
cp ui/app.py backups/session3/

# Then overwrite each file with the patched versions from this document
# (copy each "Full replacement file" section into the corresponding path)
```

## Verification Checklist

After applying all fixes:

- [ ] `engine/inference.py` — Load model, verify VRAM ~7.5GB not ~14.5GB
- [ ] `core/memory_store.py` — Call `retrieve_by_type(MemoryType.EPISODIC)`, verify no superseded memories returned
- [ ] `core/cold_storage.py` — Run resurface, verify it uses `engine.cosine_similarity` (check for missing numpy import errors = good, means it's using the engine)
- [ ] `core/reflection.py` — Generate a directive that's similar to an existing one, verify it gets rejected with "duplicate" message
- [ ] `ui/app.py` — Chat several turns, verify chatbox stays in sync with `loop.history`
- [ ] `ui/app.py` — Click "Reload from Memory" button, verify it reports directive count

## What's NOT in This Patch

These were discussed but deferred:

- **Reflection throttling** (every Nth interaction) — design decision, not a bug
- **Context window expansion** (4096 → larger) — config change, test separately  
- **`supersede_by_correction` scaling** — works fine at current memory count, revisit at 1000+
- **Backup file rename** (`interaction_loop_backup.py`) — trivial, do whenever
