import time
import re
from core.memory_store import MemoryUnit, MemoryType


# === HEURISTIC FILTER ===
# Patterns that indicate a directive is filler, not behavioral improvement.
# These replace the unreliable LLM self-scoring.

PLATITUDE_VERBS = [
    "offer", "encourage", "suggest", "provide", "share",
    "celebrate", "acknowledge", "support", "validate", "empathize",
]

PLATITUDE_NOUNS = [
    "resources", "suggestions", "encouragement", "empathy", "mindfulness",
    "well-being", "wellbeing", "self-care", "creativity", "motivation",
    "inspiration", "positivity", "relaxation", "wellness",
]

REACTIVE_PREFIXES = [
    "when the user",
    "if the user",
    "whenever the user",
    "when user",
    "if user",
]


def heuristic_quality_check(directive):
    """Score a directive using deterministic heuristics.
    
    Returns (pass: bool, reason: str).
    More reliable than a 7B model rating its own output.
    """
    d_lower = directive.lower().strip()

    # Rule 1: Too long = too vague
    if len(directive) > 120:
        return False, "too verbose (>120 chars)"

    # Rule 2: Too short = not actionable
    if len(directive) < 20:
        return False, "too short (<20 chars)"

    # Rule 3: Platitude verb + noun combo
    for verb in PLATITUDE_VERBS:
        for noun in PLATITUDE_NOUNS:
            if verb in d_lower and noun in d_lower:
                return False, f"platitude pattern: '{verb}' + '{noun}'"

    # Rule 4: Starts with reactive prefix (topic-specific, not behavioral)
    for prefix in REACTIVE_PREFIXES:
        if d_lower.startswith(prefix):
            return False, f"reactive prefix: '{prefix}'"

    # Rule 5: Contains common filler phrases
    filler_phrases = [
        "feel free", "don't hesitate", "i'm here for",
        "keep up the", "you're doing great", "take a break",
        "here to support", "here to help", "open up",
        "let me know if", "reach out", "bounce some",
    ]
    for phrase in filler_phrases:
        if phrase in d_lower:
            return False, f"filler phrase: '{phrase}'"

    # Rule 6: Must contain an actionable verb
    action_verbs = [
        "keep", "limit", "avoid", "prioritize", "ground",
        "use", "prefer", "default", "start", "end",
        "include", "omit", "structure", "format", "verify",
        "check", "confirm", "ask", "wait", "respond",
    ]
    has_action = any(verb in d_lower for verb in action_verbs)
    if not has_action:
        return False, "no actionable verb found"

    return True, "passed"


class ReflectionEngine:
    """Post-interaction recursive self-analysis at increasing depths.

    v3 changes:
    - Heuristic filter replaces LLM self-scoring (7B can't judge its own output)
    - Correction-driven directive killing: user corrections check for contradicting directives
    - Startup purge: on load, existing directives run through heuristic filter
    - Dedup threshold lowered to 0.50 to catch more semantic overlap
    - GATE 2.5: contradiction pre-check rejects directives that conflict with existing ones
      before they are stored (depth-3 coherence check only catches issues after the fact)
    """

    def __init__(self, engine, memory_store, max_depth=3,
                 max_directives=6, dedup_threshold=0.50):
        self.engine = engine
        self.memory = memory_store
        self.max_depth = max_depth
        self.max_directives = max_directives
        self.dedup_threshold = dedup_threshold
        self.directives = []
        self.directive_ids = []
        self._load_directives()

    def _load_directives(self):
        """Load existing procedural directives from memory.
        Runs heuristic filter on load — purges directives that shouldn't exist.

        Also callable from the UI to force a reload after manual pruning.
        Without this, manually deleting directives from ChromaDB leaves the
        in-memory list stale until the next full restart.
        """
        procedural = self.memory.retrieve_by_type(MemoryType.PROCEDURAL)
        procedural.sort(key=lambda m: m.salience_score, reverse=True)

        kept = []
        for m in procedural:
            passes, reason = heuristic_quality_check(m.content)
            if passes:
                kept.append(m)
            else:
                # Demote — don't delete, in case we want to inspect later
                self.memory.adjust_salience(m.id, -0.5)
                print(f"  Startup purge [{reason}]: {m.content[:60]}...")

        # Apply cap after filtering
        self.directives = [m.content for m in kept[:self.max_directives]]
        self.directive_ids = [m.id for m in kept[:self.max_directives]]

        # Demote any that exceeded cap
        if len(kept) > self.max_directives:
            for m in kept[self.max_directives:]:
                self.memory.adjust_salience(m.id, -0.3)

        print(f"Reflection engine loaded {len(self.directives)}/{self.max_directives} directives "
              f"({len(procedural) - len(kept)} purged on startup).")

    def reflect(self, user_message, assistant_response, interaction_id):
        """Run reflection pass at all depths. Returns list of new memories created.

        Depths 0 and 1 are batched into a single model call. Depth 2 and 3
        run sequentially as before (each depends on prior results).
        """
        new_memories = []

        # --- Build prompts for depth 0 and depth 1 ---
        prompt_d0 = (
            "[INST] You are a self-reflection module analyzing an interaction.\n\n"
            f"Interaction:\nUser: {user_message}\nAssistant: {assistant_response}\n\n"
            "Analyze this interaction factually:\n"
            "1. Was the response accurate and relevant?\n"
            "2. Did it address what the user actually asked?\n"
            "3. Were there any errors or misunderstandings?\n\n"
            "Be brief and specific. One paragraph. [/INST]"
        )

        recent = self.memory.retrieve(user_message, top_k=5)
        prompt_d1 = None
        if len(recent) >= 2:
            memory_context = "\n".join(
                [f"- [{m.memory_type.value}] {m.content[:150]}" for m in recent]
            )
            prompt_d1 = (
                "[INST] You are a self-reflection module analyzing patterns.\n\n"
                f"Current interaction:\nUser: {user_message}\nAssistant: {assistant_response}\n\n"
                f"Recent memories:\n{memory_context}\n\n"
                "What patterns do you notice?\n"
                "- Recurring user needs or preferences\n"
                "- Consistent strengths or weaknesses in responses\n"
                "- Emerging themes across interactions\n\n"
                "Be brief and specific. One paragraph. [/INST]"
            )

        # --- Batch generate d0 (and d1 if applicable) ---
        prompts = [prompt_d0] + ([prompt_d1] if prompt_d1 else [])
        results = self.engine.generate_batch(prompts, max_new_tokens=200, temperature=0.3)

        d0 = MemoryUnit(
            content=f"Reflection D0: {results[0]}",
            memory_type=MemoryType.REFLECTIVE,
            salience_score=0.4, confidence=0.6,
            tags=["reflection", "depth_0", "factual"],
        )
        new_memories.append(d0)

        d1 = None
        if prompt_d1:
            d1 = MemoryUnit(
                content=f"Reflection D1: {results[1]}",
                memory_type=MemoryType.REFLECTIVE,
                salience_score=0.5, confidence=0.5,
                tags=["reflection", "depth_1", "analytical"],
            )
            new_memories.append(d1)

        # --- Depth 2 and 3 sequentially ---
        d2 = self._depth_2(d0, d1)
        if d2:
            new_memories.append(d2)

        if len(self.directives) >= 2:
            d3 = self._depth_3()
            if d3:
                new_memories.append(d3)

        for mem in new_memories:
            mem.source_interaction_id = interaction_id
            self.memory.store(mem)

        return new_memories

    def check_correction_conflicts(self, correction_text):
        """Check if a user correction contradicts any active directive.
        If so, kill the directive. Called by InteractionLoop on feedback.
        
        Returns list of killed directive strings.
        """
        if not self.directives:
            return []

        correction_emb = self.engine.get_embedding(correction_text)
        killed = []

        # Check each directive for semantic similarity to the correction
        # High similarity between a correction and a directive = the directive
        # is about the behavior being corrected = kill it
        i = 0
        while i < len(self.directives):
            # Bounds check: the two lists must always be the same length.
            # If they're not, something appended to one without updating the other.
            if i >= len(self.directive_ids):
                print(f"  check_correction_conflicts: directive_ids desync at index {i} "
                      f"(directives={len(self.directives)}, ids={len(self.directive_ids)}) — skipping")
                i += 1
                continue

            directive_emb = self.engine.get_embedding(self.directives[i])
            # FIX: Use engine's shared cosine_similarity instead of manual numpy
            similarity = self.engine.cosine_similarity(correction_emb, directive_emb)

            # Threshold: 0.45 is deliberately low — corrections are phrased
            # differently from directives, so even moderate similarity is a signal
            if similarity >= 0.45:
                killed_content = self.directives[i]
                killed_id = self.directive_ids[i]
                self.memory.adjust_salience(killed_id, -0.6)
                self.directives.pop(i)
                self.directive_ids.pop(i)
                killed.append(killed_content)
                print(f"  Correction killed directive (sim={similarity:.2f}): {killed_content[:60]}...")
            else:
                i += 1

        return killed

    def _depth_2(self, d0_memory, d1_memory):
        """Prescriptive: generate behavioral directives from analysis.
        
        v3 gates:
        1. Heuristic quality filter (replaces LLM self-scoring)
        2. Semantic deduplication
        3. Directive cap with displacement
        """
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
            "Based on the analysis, should any NEW directive be added?\n\n"
            "A good directive:\n"
            '- "Keep responses under 3 sentences unless asked for detail"\n'
            '- "Ground technical explanations in practical examples"\n'
            '- "Verify factual claims before stating them confidently"\n'
            '- "Limit use of filler phrases and unsolicited advice"\n\n'
            "A BAD directive:\n"
            '- "Offer encouragement when user feels stressed" (platitude)\n'
            '- "When the user mentions Contact Front, ask about progress" (topic-specific)\n'
            '- "Be helpful and engaging" (common sense, not actionable)\n\n'
            "Rules:\n"
            "- Only propose if the analysis CLEARLY supports it\n"
            "- Must be a GENERAL behavioral rule, not topic-specific\n"
            "- Must be concise (under 100 characters ideal)\n"
            "- Must contain a concrete action verb\n"
            "- If no new directive is needed: NO_NEW_DIRECTIVE\n\n"
            "Single sentence only. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=100)

        if "NO_NEW_DIRECTIVE" in result.upper():
            return None

        # Clean up
        directive = result.strip().split("\n")[0].strip()
        directive = directive.strip('"').strip("'").strip()
        if len(directive) < 10 or len(directive) > 200:
            return None

        # === GATE 1: Heuristic quality filter ===
        passes, reason = heuristic_quality_check(directive)
        if not passes:
            print(f"  Directive rejected (heuristic: {reason}): {directive[:60]}...")
            return None

        # === GATE 2: Semantic deduplication ===
        if self._is_duplicate(directive):
            print(f"  Directive rejected (duplicate): {directive[:60]}...")
            return None

        # === GATE 2.5: Contradiction check ===
        # Depth-3 coherence only detects contradictions after the directive is
        # stored and in the live list — too late to prevent the problem.
        # This pre-check rejects directives before they are accepted.
        contradicts, which = self._is_contradiction(directive)
        if contradicts:
            print(f"  Directive rejected (contradicts '{which[:50]}'): {directive[:60]}...")
            return None

        # === GATE 3: Directive cap ===
        displaced_id = None
        if len(self.directives) >= self.max_directives:
            displaced_id = self._displace_weakest(directive)
            if displaced_id is None:
                print(f"  Directive rejected (cap full): {directive[:60]}...")
                return None

        # Accepted — create the MemoryUnit first so its ID is known, then
        # append both lists in adjacent lines. They must never be modified
        # independently; directive_ids[i] must always correspond to directives[i].
        mem = MemoryUnit(
            content=directive,
            memory_type=MemoryType.PROCEDURAL,
            salience_score=0.7,  # Fixed — earned through the filter, not self-rated
            confidence=0.7,
            tags=["reflection", "depth_2", "directive", "auto_generated"],
        )
        self.directives.append(directive)
        self.directive_ids.append(mem.id)

        # Supersede the displaced directive with the new one
        if displaced_id is not None:
            self.memory.supersede(displaced_id, mem.id)

        print(f"  Directive accepted: {directive[:60]}...")
        return mem

    def _is_duplicate(self, candidate):
        """Check semantic similarity against existing directives."""
        if not self.directives:
            return False

        candidate_emb = self.engine.get_embedding(candidate)

        for existing in self.directives:
            existing_emb = self.engine.get_embedding(existing)
            # FIX: Use engine's shared cosine_similarity instead of manual numpy
            similarity = self.engine.cosine_similarity(candidate_emb, existing_emb)
            if similarity >= self.dedup_threshold:
                return True
        return False

    def _is_contradiction(self, candidate):
        """Check if candidate logically contradicts any existing directive.

        Uses a focused LLM call constrained to 30 tokens with a binary response
        format. Temperature is set low (0.1) for deterministic output.

        A contradiction means the directives create conflicting behavioral rules
        (e.g. "keep responses brief" vs "give detailed explanations"). Mere
        difference in topic or style is not a contradiction.

        Returns:
            (True, conflicting_directive_text) if a contradiction is found
            (False, None) otherwise
        """
        if not self.directives:
            return False, None

        directive_list = "\n".join(
            f"{i+1}. {d}" for i, d in enumerate(self.directives)
        )

        prompt = (
            "[INST] Does this proposed directive logically contradict any existing directive?\n\n"
            f"Proposed: {candidate}\n\n"
            f"Existing:\n{directive_list}\n\n"
            "A contradiction means the two directives give CONFLICTING behavioral rules, "
            "e.g. 'be brief' vs 'give detailed answers', or 'use technical terms' vs 'use simple language'.\n\n"
            "Respond with exactly one of:\n"
            "CONTRADICTS: <number>\n"
            "OK\n\n"
            "If in doubt, respond OK. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=30, temperature=0.1)
        result_stripped = result.strip()

        if result_stripped.upper().startswith("CONTRADICTS"):
            parts = result_stripped.split(":", 1)
            if len(parts) > 1:
                num_str = "".join(c for c in parts[1] if c.isdigit())
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(self.directives):
                        return True, self.directives[idx]
            return True, "an existing directive"

        return False, None

    def _displace_weakest(self, new_directive):
        """If at cap, displace weakest directive. Returns displaced ID, or None if not displaced."""
        if not self.directive_ids:
            return None

        weakest_idx = None
        weakest_salience = float('inf')

        for i, mid in enumerate(self.directive_ids):
            results = self.memory.collection.get(ids=[mid])
            if results["ids"]:
                sal = results["metadatas"][0]["salience_score"]
                if sal < weakest_salience:
                    weakest_salience = sal
                    weakest_idx = i

        if weakest_idx is None:
            return None

        # New directive starts at 0.7 — must be stronger than weakest
        if 0.7 <= weakest_salience:
            return None

        old_id = self.directive_ids[weakest_idx]
        old_content = self.directives[weakest_idx]
        print(f"  Displaced directive: {old_content[:60]}...")

        self.directives.pop(weakest_idx)
        self.directive_ids.pop(weakest_idx)
        return old_id

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
            "2. Redundancies (two directives saying the same thing differently)\n"
            "3. Directives that are too vague to be actionable\n\n"
            "If all directives are coherent, respond with exactly: COHERENT\n\n"
            "If there are issues, respond in this format:\n"
            "ISSUE: [description]\n"
            "REMOVE: [number of the directive to remove, or NONE]\n\n"
            "Only recommend removing one directive per pass. "
            "Remove the less valuable of any conflicting pair. [/INST]"
        )

        result = self.engine.generate_structured(prompt, max_new_tokens=200)

        if "COHERENT" in result.upper():
            return None

        self._auto_prune(result)

        return MemoryUnit(
            content=f"Reflection D3 — Coherence check: {result}",
            memory_type=MemoryType.REFLECTIVE,
            salience_score=0.6, confidence=0.5,
            tags=["reflection", "depth_3", "coherence"],
        )

    def _auto_prune(self, coherence_result):
        """Parse coherence check and remove flagged directive."""
        try:
            for line in coherence_result.split("\n"):
                if "REMOVE:" in line.upper():
                    parts = line.upper().split("REMOVE:")
                    if len(parts) > 1:
                        num_str = ''.join(c for c in parts[1] if c.isdigit())
                        if num_str:
                            idx = int(num_str) - 1
                            if 0 <= idx < len(self.directives):
                                removed = self.directives[idx]
                                removed_id = self.directive_ids[idx]
                                self.memory.adjust_salience(removed_id, -0.5)
                                self.directives.pop(idx)
                                self.directive_ids.pop(idx)
                                print(f"  Auto-pruned directive: {removed[:60]}...")
                                return
        except (ValueError, IndexError) as e:
            print(f"  Auto-prune parse failed (non-fatal): {e}")

    def get_directives(self):
        """Return current active directives for context assembly."""
        return self.directives.copy()

    def get_directive_count(self):
        """Return current and max directive counts."""
        return len(self.directives), self.max_directives
