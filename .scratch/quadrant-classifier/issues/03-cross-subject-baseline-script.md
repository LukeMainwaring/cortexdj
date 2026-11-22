# 03: The cross-subject baseline script, kept in the repo

**Spec:** `.scratch/ml-rebuild/spec.md` (Training package and command)

**What to build:** The claim "at chance without calibration" can be re-run from the repo. The script that produced it lives only on the local branch `refs/heads/prototype/ml-rebuild-baseline` (commit bcfe9dd); bring it into the training package as one standalone file that scores logistic regression on differential entropy under leave-one-subject-out and within-subject by trial. It is evidence, not part of the training path: the training code gets no leave-one-subject-out mode.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] The script is one file in the training package, run directly with a data directory argument, and prints the table ml-rebuild 04 recorded
- [x] It reads labels through ticket 01's ratings loader, never from the files' label array
- [x] It is the only file that imports scikit-learn, which stays a transitive dependency; its docstring says so and says why the file is kept
- [x] It passes mypy strict and the import-linter contracts; it has no test and nothing imports it
- [x] Docs call it the cross-subject baseline script, never a "baseline run" (that term is the eval tier's in the glossary)
