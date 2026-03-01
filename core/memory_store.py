import json
import time
import math
import uuid
import chromadb
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
    """Persistent semantic memory with vector search and salience decay.
    
    v3 changes:
    - supersede() method to mark individual memories as superseded
    - supersede_by_correction() to find and mark memories that contradict a correction
    - Breaks hallucination reinforcement loop: wrong answers stored as episodic
      memories get superseded when the user corrects the record
    """

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
                similarity = 1 - results["distances"][0][i]
                blended_score = (similarity * similarity_weight) + (mem.salience_score * salience_weight)
                scored_memories.append((mem, blended_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored_memories[:top_k]]

    def retrieve_by_type(self, memory_type: MemoryType, limit=50):
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

    def supersede(self, memory_id, superseded_by_id):
        """Mark a memory as superseded by another memory.
        Superseded memories are excluded from retrieval."""
        results = self.collection.get(ids=[memory_id])
        if not results["ids"]:
            return
        metadata = results["metadatas"][0]
        metadata["superseded_by"] = superseded_by_id
        metadata["salience_score"] = max(0.0, metadata["salience_score"] - 0.3)
        self.collection.update(ids=[memory_id], metadatas=[metadata])

    def supersede_by_correction(self, correction_text, correction_id, similarity_threshold=0.45):
        """Find memories that contradict a correction and mark them superseded.
        
        This breaks the hallucination reinforcement loop:
        1. Model hallucinates wrong answer
        2. Wrong answer stored as episodic memory  
        3. Next query retrieves the wrong episodic memory
        4. Model reinforces the hallucination
        
        By superseding memories similar to the correction topic, we prevent
        step 3 from retrieving the wrong answer.
        
        Only targets episodic and semantic memories (not procedural/reflective).
        Only targets non-correction memories (don't supersede other corrections).
        """
        if self.engine is None:
            return []

        correction_emb = self.engine.get_embedding(correction_text)

        # Get all memories with embeddings
        all_data = self.collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        superseded = []
        for i in range(len(all_data["ids"])):
            metadata = all_data["metadatas"][i]
            document = all_data["documents"][i]
            mem_id = all_data["ids"][i]

            # Skip if already superseded
            if metadata.get("superseded_by", ""):
                continue

            # Skip procedural and reflective — corrections target facts, not directives
            if metadata["memory_type"] in ("procedural", "reflective"):
                continue

            # Skip other corrections — don't supersede user ground truth
            tags = json.loads(metadata.get("tags", "[]"))
            if "user_correction" in tags:
                continue

            # Skip if this IS the correction we're processing
            if mem_id == correction_id:
                continue

            # Check semantic similarity
            if all_data["embeddings"] is not None and all_data["embeddings"][i] is not None:
                similarity = self.engine.cosine_similarity(
                    correction_emb, all_data["embeddings"][i]
                )
                if similarity >= similarity_threshold:
                    # This memory is about the same topic as the correction
                    # and is NOT a correction itself — it's likely the wrong answer
                    self.supersede(mem_id, correction_id)
                    superseded.append({
                        "id": mem_id,
                        "content": document[:80],
                        "similarity": float(similarity),
                        "type": metadata["memory_type"],
                    })

        if superseded:
            print(f"  Correction superseded {len(superseded)} memories:")
            for s in superseded:
                print(f"    [{s['type']}] sim={s['similarity']:.2f} | {s['content']}...")

        return superseded

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
        superseded_count = 0
        for m in all_memories["metadatas"]:
            t = m["memory_type"]
            types[t] = types.get(t, 0) + 1
            saliences.append(m["salience_score"])
            if m.get("superseded_by", ""):
                superseded_count += 1

        return {
            "total": total,
            "by_type": types,
            "avg_salience": sum(saliences) / len(saliences),
            "min_salience": min(saliences),
            "max_salience": max(saliences),
            "superseded": superseded_count,
        }

    def _update_access(self, memory: MemoryUnit):
        self.collection.update(
            ids=[memory.id],
            metadatas=[memory.to_metadata()],
        )
