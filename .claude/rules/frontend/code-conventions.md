---
paths:
  - "frontend/components/**/*.{ts,tsx}"
  - "frontend/app/**/*.{ts,tsx}"
  - "frontend/hooks/**/*.{ts,tsx}"
  - "frontend/api/hooks/**/*.{ts,tsx}"
  - "frontend/lib/**/*.{ts,tsx}"
---

# Frontend Patterns

TypeScript/Next.js conventions for the cortexdj frontend.

## Imports

- Use the `@/` path alias for imports that cross top-level directories
  (e.g., `import { cn } from "@/lib/utils"`,
  `import { useSimilarTracks } from "@/api/hooks/sessions"`).
- Relative imports are fine for files in the same directory or in a direct
  subdirectory grouping within `components/` (e.g., `./ui/button`,
  `./elements/tool-call`, `./sidebar-history-item`). This matches the
  Vercel chatbot template pattern the codebase descends from.

## UI Components

- Use shadcn/ui components from `components/ui/` instead of raw HTML elements:
  - `<Button>` over `<button>` — provides consistent focus rings, disabled
    states, and cursor styles.
  - `<Input>` over `<input>` — except hidden file inputs
    (`type="file" className="hidden"`) and non-text inputs like sliders
    (`type="range"`), which are fine as raw elements.
  - `<Textarea>` over `<textarea>`.
  - `<Skeleton>` over `<div className="animate-pulse ...">` for loading
    placeholders.
  - `<Tooltip>` over `title=` attributes on interactive elements (buttons,
    icon buttons). Native `title=` is acceptable on text elements for
    truncation hints.
  - `<Separator>` for standalone visual dividers. Border classes (`border-b`,
    `border-t`) are fine when the border is part of a container's layout.
  - `<AlertDialog>` for confirmation dialogs, `<DropdownMenu>` for context
    menus, `<Collapsible>` for expandable sections, `<Sheet>` for slide-out
    panels, `<Tabs>` for tabbed groupings.
  - `<Card>` is available in `components/ui/`, but the project predominantly
    uses raw `border-border bg-card` divs for card-shaped surfaces when the
    finer color/layout control matters (see `session-list-panel.tsx`,
    `retrieved-tracks-panel.tsx`). Either pattern is fine; reach for `<Card>`
    when a section truly is a card-and-nothing-else.
- When a shadcn component's default variant matches your needs, don't rewrite
  the styles — just use the variant. Override with `className` only for
  styles the variant doesn't cover.
- Do not manually edit files in `components/ui/` unless adding a new shadcn
  component or customizing an existing variant.

## Data fetching

- Wrap generated API client calls in TanStack Query hooks (e.g., `useSessionSegments` in
  `api/hooks/sessions.ts`) rather than calling the generated client directly from
  components; keep the hooks under `api/hooks/`.
- Never hand-edit `api/generated/` — it is regenerated from the backend OpenAPI
  schema via `pnpm -C frontend generate-client`.

## Code Style

- Use kebab-case for filenames (e.g., `session-visualization.tsx`,
  `multimodal-input.tsx`).
- Colocate types with their component unless shared across multiple files.
- Use `useCallback` and `memo` where the current codebase already does
  (`messages.tsx`, `session-list-panel.tsx`, etc. wrap their components in
  `memo` with a custom equality fn). Don't add new memoization speculatively
  — Next.js 16 ships with React 19, and once the React Compiler is enabled
  most manual `useCallback`/`useMemo` becomes redundant or counterproductive.
- Use `cn()` from `@/lib/utils` for conditional class merging.
