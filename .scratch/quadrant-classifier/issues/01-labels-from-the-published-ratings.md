# 01: Labels from the published ratings

**Spec:** `.scratch/ml-rebuild/spec.md` (Data and labels)

**What to build:** Valence and arousal for a DEAP participant come from DEAP's published ratings file, kept beside the EEG files, and never from the label array inside the mirror's files, which is altered in 439 trials. A caller asks for a participant's ratings and gets 40 trials of valence and arousal in the same order as the EEG. If the ratings do not line up with the EEG file, loading fails instead of producing a number. Nothing reads the new labels yet; ticket 02 is the first caller.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] A participant's ratings are read from the ratings file in a data directory passed as an argument, joined on participant and experiment id (a file's trial index `e` is experiment id `e + 1`)
- [ ] Loading raises, naming the participant, when the ratings file is missing
- [ ] Loading raises when the ratings' dominance and liking differ from the EEG file's own label array: that comparison is the check that the join is aligned
- [ ] "High" is one function of a rating: at or above 5
- [ ] The EEG the DEAP loader returns is unchanged, and the retrieval tests pass untouched
- [ ] A shared test builder writes one made-up participant in DEAP's shape (the pickle and a matching ratings file) under a temporary directory; no test reads `backend/data/`
- [ ] `backend/data/DEAP_SETUP.md` says the mirror's valence and arousal are altered, drops "the same preprocessed data" and the "~24/76" split, and adds the ratings-file step with its source and checksum (ml-rebuild 09 has both)
