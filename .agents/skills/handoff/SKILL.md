---
name: handoff
description: Use to create/update a handoff document (`HANDOFF.md`) at the end of a session, OR to resume work from a previous handoff, transcript, or session summary at the start of a session.
---

# Handoff & Session Resume

## Purpose
Manage the complete lifecycle of session handoffs:
1. **Resume**: Reconstruct context and task states from a prior session's handoff or transcript before editing files.
2. **Handoff**: Generate a clean, detailed `HANDOFF.md` at the end of the session to transition work smoothly to the next agent.

---

## PART 1: Resume Workflow (Subentering Agent)
When continuing from a previous session, the agent must reconstruct the context, verify the files, and start from the first unfinished step without duplicating work.

### 1. Locate and Read Context
- Inspect the workspace for `HANDOFF.md` (root directory) or session logs (e.g., in `.system_generated/logs/` or `appDataDir`).
- Prefer explicit paths provided by the user.
- Read the entire handoff or the relevant slices of the prior transcript before modifying files or running commands.

### 2. Reconstruct Context & Validate Workspace
- Summarize the session goal.
- Extract decisions, constraints, and user preferences.
- Check `git status` and inspect modified files to verify their actual state. If the worktree is dirty, preserve unrelated changes and do not overwrite them without permission.
- If there are conflicts between what the transcript/handoff claims and what is actually in the files, trust the files and note the mismatch.
- Document any user-parked or deferred tasks (e.g., "skip", "park", "leave out") and preserve them.

### 3. Extract and Classify Tasks
Classify all tasks into:
- `DONE`: Completed and verified (with concrete references/evidence).
- `PARTIALLY DONE`: Started but missing implementation, verification, or user approval.
- `NOT DONE`: Unstarted or only discussed.

*Every status line must include concrete evidence (e.g., file paths, line ranges, or command outputs).*

### 4. Required Resume Report Shape
Before executing new edits or commands, output the following status:
```markdown
## Brief context summary
- **Goal**: <prior session goal>
- **Handoff/Source reviewed**: <file paths, transcript lines>
- **Current workspace check**: <git status summary and touched-file refs>
- **Mismatches**: <claim vs actual files, or "none found">
- **User deferrals**: <deferred scopes and reopen conditions, or "none found">
- **Stopping point**: <last command, edit, or failure with evidence>

## Task status breakdown
- **DONE**: <completed task> - evidence: <refs>; verification: <test/tool refs or "not recorded">
- **PARTIALLY DONE**: <started task> - evidence: <refs>; missing: <refs>
- **NOT DONE**: <unstarted task> - evidence: <refs>

## Clear next action
- **Next**: <first unfinished step to take now>
- **Blocked**: <no | yes - reason>
```

---

## PART 2: Handoff Generation (Exiting Agent)
When completing a session, or when asked to perform a handoff, generate or update `HANDOFF.md` in the project root.

### 1. Process
1. **Locate**: Check if `HANDOFF.md` exists. If so, read it to preserve relevant ongoing context.
2. **Compile**: Structure the document with the following mandatory sections:
   - **Goal**: What the project or current task is trying to accomplish.
   - **Current Progress**: Concrete tasks completed in the current session.
   - **What Worked**: Approaches, designs, or implementations that succeeded.
   - **What Didn't Work**: Approaches or experiments that failed (crucial so the next agent doesn't repeat the same mistakes).
   - **Next Steps**: Detailed, actionable, and structured next steps for the next session.
   - **Suggested Skills**: Specific skills the next agent should invoke or be aware of.
3. **References**: Do not duplicate large plans or PRDs; reference them by file links (e.g., `[file.md](file:///...)`).
4. **Security**: Redact all sensitive information (API keys, credentials, PII).
5. **Save**: Save to `HANDOFF.md` in the project root.
6. **Notify**: Report the absolute file path of the saved `HANDOFF.md` to the user.

---

## Guardrails
- **No premature editing**: Do not make changes or run fix commands during the resume phase until the resume report is generated.
- **Trust the workspace**: Real files and `git status` override any claims made in transcripts or handoffs.
- **Evidence is mandatory**: Do not label tasks as `DONE` or `PARTIALLY DONE` without specifying concrete files/lines or command results.
- **Preserve local changes**: Never reset, revert, or discard dirty files unless the user explicitly requests it.

