#!/usr/bin/env bash
# Report what changed upstream in the skills vendored from mattpocock/skills
# since the commit pinned in .agents/skills/VENDORED.md, and which upstream
# skills are neither vendored nor listed as skipped. Prints "up to date" when
# both are empty. Read-only: it never touches the vendored copies.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$repo_root/.agents/skills/VENDORED.md"
upstream_url="https://github.com/mattpocock/skills"

pinned="$(sed -n 's/^Upstream commit: *//p' "$manifest")"
[ -n "$pinned" ] || { echo "error: no 'Upstream commit: <sha>' line in $manifest" >&2; exit 1; }
section_entries() { # print the "- bucket/name" entries under one "## heading"
  awk -v h="## $1" '$0 == h {on=1; next} /^## / {on=0} on && /^- / {print $2}' "$manifest"
}
vendored="$(section_entries Vendored)"
skipped="$(section_entries "Not vendored")"

clone="$(mktemp -d)"
trap 'rm -rf "$clone"' EXIT
git clone --quiet --filter=blob:none "$upstream_url" "$clone"
head="$(git -C "$clone" rev-parse --short HEAD)"
git -C "$clone" cat-file -e "$pinned^{commit}" 2>/dev/null \
  || { echo "error: pinned commit $pinned is not in $upstream_url" >&2; exit 1; }
version="$(sed -n 's/^  "version": "\(.*\)",$/\1/p' "$clone/package.json" | head -1)"
echo "upstream: $head (package version $version), pinned: $pinned"

changed=0
for entry in $vendored; do
  log="$(git -C "$clone" log "$pinned..HEAD" --format='  %h %ad %s' --date=short -- "skills/$entry")"
  [ -n "$log" ] || continue
  [ $changed -eq 1 ] || { echo; echo "Changed since pin:"; changed=1; }
  echo "$entry"
  echo "$log"
done

new=0
for dir in "$clone"/skills/engineering/*/ "$clone"/skills/productivity/*/; do
  entry="${dir#"$clone"/skills/}"; entry="${entry%/}"
  grep -qx "$entry" <<<"$vendored"$'\n'"$skipped" && continue
  [ $new -eq 1 ] || { echo; echo "New upstream skills (not vendored, not skipped):"; new=1; }
  desc="$([ -f "$dir/SKILL.md" ] && sed -n 's/^description: *//p' "$dir/SKILL.md" | head -1 || echo "(no SKILL.md)")"
  echo "$entry: $desc"
done

[ $changed -eq 1 ] || [ $new -eq 1 ] || echo "up to date"
