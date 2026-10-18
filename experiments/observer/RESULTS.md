# Results: observer experiment, stopped early

**Status:** stopped by Keith on 2026-10-18 at 17:40, after 6 of the 24 pre-registered runs. A seventh run (PEER seed 2) was killed partway through. It is kept in `aborted/` and excluded from the analysis.
**Data:** `/mnt/vault/projects/cultivated-learning/observer/runs/`, `analysis.json`.

## Q1 (primary): no answer

The pre-registered rule needs 8 SELF_PLUS/PEER pairs, and only 1 finished. The table prints FAIL because `verdict()` runs on whatever pairs exist. That label carries no weight here; with one pair it is not a verdict.

The early stop is not the main problem. **The test could not have passed on this setup.**

| Arm | Runs | M1 | Lists | Question | Length |
|---|---|---|---|---|---|
| SELF | 2 | 0.030 | 0.033 | 0.048 | 0.008 |
| SELF_PLUS | 2 | 0.003 | 0.008 | 0.000 | 0.000 |
| PEER | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| OTHER | 1 | 0.011 | 0.000 | 0.032 | 0.000 |

The behaviours being corrected barely occur, even before the corrections. Across all arms, prompts 1–5 drew one trailing question in 30 replies; the median reply is 34–51 words; list replies are 0–5 per run. The Sable system prompt already forbids lists and long answers, and Mistral Small 2501 on this stack obeys it. The bar asked PEER to beat SELF_PLUS by 5 points, but SELF_PLUS sits at 0.3%, so the bar is unreachable.

**The design error, mine:** the endpoints were chosen from the failures The Ceiling Holds reported, without first measuring how often those failures occur in this setup. The smoke runs covered 9 prompts, and no base rate was computed from them. A pre-registration needs a base-rate check before the bar is set.

## Q5: holds

The reflective share was 50.6–51.2% in every arm, with a gap of 0.6 points between arms, whoever wrote the reflections. The share is set by the reflection schedule, not by the author or the model. The 48% in The Ceiling Holds was the schedule.

## Not pre-registered: what the authors wrote

This is an observation, not a test. I read it unblinded, with 1–2 runs per arm.

The final directive sets differ visibly:
- **SELF** wrote directives that ask the user questions in both runs, after two corrections against ending on a question.
  - s1: "Propose more detailed follow-up questions to deepen understanding of user needs."
  - s2: "Ask clarifying questions…", "Prompt user to confirm their current knowledge level…", "Encourage user to specify desired format or style upfront."
- **SELF_PLUS** wrote one borderline example ("Confirm user instructions before proceeding with tasks"), and also "Reiterate user corrections promptly…".
- **PEER** wrote none, and wrote "Acknowledge and implement user corrections promptly."
- **OTHER** wrote none, and encoded the corrections themselves: "Adhere strictly to formatting constraints, such as avoiding trailing questions after explicit instructions to stop."

Those directives barely reached behaviour: SELF's trailing-question rate was 4.8%. In this setup the system prompt dominates, so the self-model is not the lever on replies. Who writes the self-model may change what it says, but that hardly matters when nothing it says drives behaviour.

## What a real test needs

1. **Measure base rates first.** Run the shipped design and keep only endpoints whose baseline rate leaves room to fall.
2. **Make the self-model matter.** Run without the system prompt's hard rules, so directives carry the behaviour. Or choose behaviours the system prompt does not already govern.
3. **Consider a different primary endpoint: the self-model itself.** Count directives that contradict a standing user correction, graded blind by a third model (J2). That is where the arms looked different.

That is a new pre-registration, written before its runs.
