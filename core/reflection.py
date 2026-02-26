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
        """Load existing procedural directives from memory."""
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
