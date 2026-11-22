# 08: Where training code lives, the checkpoint, and how Modal fits

**Type:** grilling
**Status:** open
**Blocked by:** 03, 07

## Question

Where does training code live so the `cortexdj.ml` inference kernel stays import-clean, and how do training and inference share one preprocessing path without copying it? What is in a checkpoint (weights, the preprocessing settings it was trained with, metrics) so a mismatch like `after-port 01` fails loudly at load, and how does a checkpoint get from a run to the app? Which is the default command, the local single-fold run or the Modal full run, and how few settings can the training function take so there is one code path?
