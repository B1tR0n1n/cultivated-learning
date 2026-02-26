import time
from core.memory_store import MemoryUnit, MemoryType


class ConsolidationEngine:
    """Distills fading episodic memories into durable semantic memories."""

    def __init__(self, engine, memory_store, salience_threshold=0.4, min_cluster=2):
        self.engine = engine
        self.memory = memory_store
        self.salience_threshold = salience_threshold
        self.min_cluster = min_cluster

    def consolidate(self):
        """Run a consolidation pass. Returns list of new semantic memories created."""
        # Get all episodic memories
        episodic = self.memory.retrieve_by_type(MemoryType.EPISODIC)

        if len(episodic) < self.min_cluster:
            print(f"Consolidation: only {len(episodic)} episodic memories, need {self.min_cluster}. Skipping.")
            return []

        # Split into fading (candidates for consolidation) and fresh
        fading = [m for m in episodic if m.salience_score < self.salience_threshold]
        fresh = [m for m in episodic if m.salience_score >= self.salience_threshold]

        if len(fading) < self.min_cluster:
            print(f"Consolidation: only {len(fading)} fading memories (below {self.salience_threshold}). Skipping.")
            return []

        print(f"Consolidation: {len(fading)} fading episodic memories found. Distilling...")

        # Feed fading memories to the LLM for distillation
        memory_text = "\n\n".join(
            [f"[Episode {i+1}] {m.content}" for i, m in enumerate(fading)]
        )

        prompt = (
            "[INST] You are a memory consolidation module. Your job is to extract "
            "lasting knowledge from a set of interaction episodes.\n\n"
            f"Episodes:\n{memory_text}\n\n"
            "Extract the durable facts, preferences, and patterns from these episodes. "
            "Ignore transient details (greetings, small talk, one-time questions).\n\n"
            "Format: Return one fact per line. Each fact should be a standalone statement "
            "that would be useful to remember long-term. If there are no lasting facts, "
            "respond with exactly: NOTHING_TO_CONSOLIDATE\n\n"
            "Be concise. Maximum 5 facts. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=300)

        if "NOTHING_TO_CONSOLIDATE" in result.upper():
            print("Consolidation: no lasting knowledge extracted.")
            return []

        # Parse facts into semantic memories
        new_memories = []
        facts = [line.strip() for line in result.strip().split("\n") if line.strip()]

        for fact in facts[:5]:
            # Clean up common LLM formatting
            fact = fact.lstrip("0123456789.-) ").strip()
            if len(fact) < 10:
                continue

            semantic = MemoryUnit(
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                salience_score=0.6,
                confidence=0.7,
                tags=["consolidated", "auto_generated"],
            )
            self.memory.store(semantic)
            new_memories.append(semantic)

        # Mark consolidated episodic memories as superseded
        if new_memories:
            for mem in fading:
                self.memory.adjust_salience(mem.id, -0.2)

        print(f"Consolidation complete: {len(new_memories)} semantic memories created, "
              f"{len(fading)} episodic memories demoted.")
        return new_memories

    def get_consolidation_candidates(self):
        """Preview what would be consolidated without doing it."""
        episodic = self.memory.retrieve_by_type(MemoryType.EPISODIC)
        fading = [m for m in episodic if m.salience_score < self.salience_threshold]
        return fading
