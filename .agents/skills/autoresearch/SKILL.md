---
name: autoresearch
description: Stage an autoresearch batch — report the champion state and print the /goal line that starts the loop.
argument-hint: "[N — experiments in the batch, default 10]"
disable-model-invocation: true
---

Stage a batch of autoresearch experiments for the EEG emotion classifier. N is the argument if one was passed, else 10.

1. Read `backend/autoresearch/program.md` — the experiment contract you follow once the batch starts.
2. Report the current state: the champion `avg_macro_f1` from `backend/autoresearch/experiments/best.json` plus the last few rows of `experiments/experiments.jsonl`. If `experiments/` doesn't exist yet, report "no runs yet in this checkout" (chance baseline is 0.5).
3. Print this goal line, with N substituted, for the user to run — then stop and wait:

   ```
   /goal N new experiment rows appended to backend/autoresearch/experiments/experiments.jsonl this session — after each run the agent reports the new row's metric, status, and keep/revert decision
   ```

   (Codex session instead? Same condition via Codex's `/goal`.)

Staging launches nothing: experiments burn real Modal GPU time, and the batch begins only when the user sets the goal or explicitly says go. Your job ends at the printed goal line.
