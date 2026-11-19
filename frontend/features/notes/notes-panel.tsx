"use client";

import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Note {
  content: string;
  created_at: string;
  id: string;
  title: string;
}

// The `list_notes` tool returns `{ notes: [...] }`, but a tool result can reach
// the UI either as the parsed object or as a JSON string (depending on how the
// backend serialized it). Parse defensively and treat anything off-shape as
// "nothing to render" rather than throwing inside a render.
function parseNotes(output: unknown): Note[] | null {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }

  if (
    !value ||
    typeof value !== "object" ||
    !Array.isArray((value as { notes?: unknown }).notes)
  ) {
    return null;
  }

  const notes: Note[] = [];
  for (const raw of (value as { notes: unknown[] }).notes) {
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const candidate = raw as Record<string, unknown>;
    if (
      typeof candidate.id === "string" &&
      typeof candidate.title === "string" &&
      typeof candidate.content === "string" &&
      typeof candidate.created_at === "string"
    ) {
      notes.push({
        content: candidate.content,
        created_at: candidate.created_at,
        id: candidate.id,
        title: candidate.title,
      });
    }
  }
  return notes;
}

function formatCreatedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

const PureNotesPanel = ({ output }: { output: unknown }) => {
  const notes = parseNotes(output);

  if (notes === null) {
    return (
      <Card className="text-muted-foreground p-3 text-xs">
        Couldn't read the notes returned by this tool.
      </Card>
    );
  }

  if (notes.length === 0) {
    return (
      <Card className="text-muted-foreground p-3 text-xs">No notes yet.</Card>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {notes.map((note) => (
        <Card key={note.id}>
          <CardHeader className="gap-1 p-3 pb-0">
            <CardTitle className="text-sm leading-snug">{note.title}</CardTitle>
            <span className="text-muted-foreground text-xs">
              {formatCreatedAt(note.created_at)}
            </span>
          </CardHeader>
          <CardContent className="p-3 pt-2">
            <p className="text-foreground text-sm wrap-break-word whitespace-pre-wrap">
              {note.content}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export const NotesPanel = memo(PureNotesPanel);
