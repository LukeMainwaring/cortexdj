---
paths:
  - "frontend/components/**/*.{ts,tsx}"
  - "frontend/features/**/*.{ts,tsx}"
  - "frontend/app/**/*.{ts,tsx}"
  - "frontend/hooks/**/*.{ts,tsx}"
  - "frontend/api/hooks/**/*.{ts,tsx}"
  - "frontend/lib/**/*.{ts,tsx}"
---

# Frontend rules

What the frontend code can't show you before you write it. Everything it does show is a review
check in `REVIEW.md`, and lint enforces the import conventions and bans a raw `button`, `input`,
`textarea`, or `hr`.

- **A component missing from `components/ui/`:** add it with the shadcn CLI (`shadcn add
  <component>`) rather than hand-writing it.
- **A call to the generated API client:** wrap it in a TanStack Query hook; copy `useThreads` in
  `api/hooks/threads.ts`.
