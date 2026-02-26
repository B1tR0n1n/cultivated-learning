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
            memories.append(mem)
        return memories

    def adjust_salience(self, memory_id, delta):
        results = self.collection.get(ids=[memory_id])
        if not results["ids"]:
            return
        metadata = results["metadatas"][0]
        metadata["salience_score"] = max(0.0, min(1.0, metadata["salience_score"] + delta))
        self.collection.update(ids=[memory_id], metadatas=[metadata])

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
