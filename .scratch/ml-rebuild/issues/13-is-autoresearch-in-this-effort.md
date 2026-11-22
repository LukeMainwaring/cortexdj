# 13: Is an autoresearch loop in this effort?

**Type:** grilling
**Status:** open

## Question

Once `train-classifier` and the single number exist, does this effort's spec include a `/goal`-style loop that changes the training recipe and reruns, or does that wait for the bake-off effort? The model is a linear layer with fixed, untuned settings (tickets 07 and 12), so the loop would have the weight decay, learning rate, epochs and the feature choices to move, and `train_participant`'s `recipe` argument is the opening left for it (ticket 08). If it is in: what it may change, what stops it tuning on the same folds the headline number is reported on, and where its runs are recorded. If it is out: one line in the map's Out of scope.

## Comments

- 2026-11-22: graduated from the map's fog when ticket 12 resolved and no other decision was left open.
