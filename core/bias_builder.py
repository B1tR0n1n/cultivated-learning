import re
from core.memory_store import MemoryType


# Static phrases to suppress unconditionally — common instruct-isms that
# Mistral-Instruct produces even when the system prompt tells it not to.
BASE_SUPPRESSIONS = [
    "Certainly!", "Certainly,",
    "Sure!", "Sure,",
    "Of course!", "Of course,",
    "Absolutely!", "Absolutely,",
    "Great question",
    "I'd be happy to",
    "I would be happy to",
    "As an AI",
    "As a language model",
    "I hope this helps",
    "Feel free to",
    "Don't hesitate",
    "I'm here to help",
    "Happy to help",
]

# Regex patterns for freetext/legacy user_correction memories.
# Matched against lowercased content after stripping the "CORRECTION:" prefix.
_FREETEXT_PATTERNS = [
    r"stop saying\s+[\"']?(.+?)[\"']?[.!]?$",
    r"don't (?:use|say)\s+[\"']?(.+?)[\"']?[.!]?$",
    r"never say\s+[\"']?(.+?)[\"']?[.!]?$",
    r"stop using\s+[\"']?(.+?)[\"']?[.!]?$",
    r"avoid saying\s+[\"']?(.+?)[\"']?[.!]?$",
    r"avoid using\s+[\"']?(.+?)[\"']?[.!]?$",
    r"stop being so\s+[\"']?(.+?)[\"']?[.!]?$",
]

# Structured tag → list of prefixes that may have been prepended when storing.
# Multiple prefixes per tag because the 20-template UI uses several templates
# that map to the same suppression category.
_STRUCTURED_PREFIXES = {
    "suppress_phrase":   ["Stop saying ", "Don't ", "Never "],
    "suppress_behavior": ["Stop being ", "Less "],
}


class BiasBuilder:
    """Builds a logit bias map from static suppressions and user corrections.

    Two extraction paths, combined and deduplicated before tokenization:

    1. Structured — memories tagged suppress_phrase or suppress_behavior.
       Target is extracted by stripping the known prefix from content.
       No regex needed; the prefix was enforced at storage time.

    2. Freetext/legacy — memories tagged user_correction without a structured
       suppression tag. Target is extracted via regex patterns that match
       natural-language instructions like "stop saying X" or "never use X".

    No LLM calls — this runs on every generation and must be fast.
    """

    def __init__(self, engine, memory_store):
        self.engine = engine
        self.memory = memory_store

    def build_bias_map(self):
        """Build and return a {token_id: -10.0} bias map.

        Returns:
            dict mapping token_id (int) to -10.0 float
        """
        phrases = list(BASE_SUPPRESSIONS)
        seen = {p.lower() for p in phrases}

        mems = self.memory.retrieve_by_type(MemoryType.SEMANTIC, limit=100)

        # Path 1: structured memories with explicit suppression tags
        for mem in mems:
            for tag, prefixes in _STRUCTURED_PREFIXES.items():
                if tag in mem.tags:
                    target = self._extract_from_structured(mem.content, prefixes)
                    if target and target.lower() not in seen:
                        phrases.append(target)
                        seen.add(target.lower())

        # Path 2: freetext / legacy user_correction memories
        for mem in mems:
            if "user_correction" not in mem.tags:
                continue
            # Skip structured ones — already handled above
            if any(t in mem.tags for t in _STRUCTURED_PREFIXES):
                continue
            target = self._extract_from_freetext(mem.content)
            if target and target.lower() not in seen:
                phrases.append(target)
                seen.add(target.lower())

        return self.engine.suppress_tokens(phrases)

    def _extract_from_structured(self, content, prefixes):
        """Try each known prefix and strip the first one that matches.

        Args:
            content:  Memory content, e.g. "Don't use filler phrases"
            prefixes: List of candidate prefixes, e.g. ["Stop saying ", "Don't ", "Never "]

        Returns:
            Target phrase string, or None if no prefix matched.
        """
        for prefix in prefixes:
            if content.lower().startswith(prefix.lower()):
                return content[len(prefix):].strip()
        return None

    def _extract_from_freetext(self, text):
        """Apply regex patterns to extract a suppression target from freetext.

        Args:
            text: Raw correction memory content
                  (e.g. "CORRECTION: stop saying Certainly!")

        Returns:
            Extracted phrase string, or None if no pattern matched.
        """
        text = re.sub(r"^CORRECTION:\s*", "", text, flags=re.IGNORECASE).strip()
        text_lower = text.lower()
        for pattern in _FREETEXT_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                return m.group(1).strip()
        return None
