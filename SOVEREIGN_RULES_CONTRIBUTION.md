# Sovereign Engine Development Rules

These rules override everything else. If we break them, we go back.

### Rule 1: Master Before Move
> Do NOT move to the next phase until the current phase is **verified with numbers**.
> "It works" is not proof. "MSE dropped from X to Y on a random signal" is proof.

### Rule 2: No Over-Engineering
> Every feature must have a **measurable reason** to exist.
> If we can't explain why a feature improves the benchmark, it gets deleted.

### Rule 3: No Feature Bloat
> One feature per phase. Test it. Verify it. Then move on.
> The v11 disaster happened because we added 10 features at once and couldn't debug any of them.

### Rule 4: Stay on the Current Step
> Do NOT think about the final swarm goal while working on Phase 4.
> Each phase has its own success metric. Hit that metric. Nothing else matters.

### Rule 5: No Rushing
> If a phase takes 5 sessions to get right, it takes 5 sessions.
> Cutting corners now creates bugs that explode later (see: v11 gradient corruption).

### Rule 6: Document Everything
> Every phase must record: what we tried, what worked, what failed, and why.
> This journal is maintained in `CONTRIBUTOR_JOURNAL.md`.

### Rule 7: Ask, Don't Assume
> If something is unclear, ASK before coding.
> Wrong assumptions waste hours. A question takes 10 seconds.

### Rule 8: No Overhype
> We will not say "breakthrough" unless the numbers prove it.
> We will not say "it works" unless an ablation confirms it.
> We will not compare to Transformers until we have standard benchmark results.

### Rule 9: Brutal Honesty
> Point out every flaw immediately. Do not hide problems to "fix later."
> If the architecture is fundamentally wrong, say so. Better to know at Phase 4 than Phase 9.

### Rule 10: Smallest Possible Change
> When debugging, change ONE thing at a time.
> When adding features, add ONE thing at a time.
> "I changed 3 things and now it works" means you don't know which one fixed it.
