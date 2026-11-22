# 12: What a re-seeded session shows at about 0.57

**Type:** grilling
**Status:** open
**Blocked by:** 08

## Question

After `train-classifier` and a re-seed, every segment's arousal and valence scores are a fold model's held-out output (ticket 08), from a linear layer that is right about 57% of the time. Are the stored 0 to 1 scores the sigmoid of the logits as they come, or do they need calibrating so they are not all bunched near 0.5 or pushed to the ends? What should a session's quadrant path and trajectory analytics look like in the app at that accuracy, and does anything the app or the assistant says about a session need to state how reliable it is? Does the app contract (scores 0 to 1, a quadrant per segment) hold as it is?

## Comments

- 2026-11-22: graduated from the map's fog when ticket 08 settled how seeding gets held-out predictions. No model has been trained yet, so the spread of real scores is unknown; a first `train-classifier` run, or the baseline on `refs/heads/prototype/ml-rebuild-baseline`, would show it.
