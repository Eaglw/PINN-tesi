# PINN Wiki Schema

This file defines the protocols for the LLM-managed Wiki within this vault.

## Directory Structure
- `Reference/`: Raw source materials (PDFs, scripts, data). Immutable.
- `Wiki/`: The LLM-managed knowledge layer.
  - `00_Index.md`: High-level navigation and catalog.
  - `01_Log.md`: Chronological history of actions.
  - `Literature/`: Detailed summaries of individual sources from `Reference/`.
  - `Topics/`: Synthesis of physical and mathematical concepts.
  - `Methods/`: Technical and algorithmic implementations.
  - `Systems/`: Domain-specific analysis (Heat2D, CSTR, etc.).

## Maintenance Protocols

### 1. Ingestion Workflow
When a new source is added to `Reference/`:
1. **Read**: Analyze the source content.
2. **Log**: Add entry to `Wiki/01_Log.md` (e.g., `## [YYYY-MM-DD] ingest | Source Name`).
3. **Summarize**: Create a page in `Wiki/Literature/Source_Name.md` using the **Literature Template**.
4. **Integrate**: Use surgical `replace` calls to update existing pages in `Wiki/` to reflect new information.
5. **Index**: Ensure the new literature page and any new topic/method pages are listed in `Wiki/00_Index.md`.

### 2. Surgical Updates
- Never overwrite a consolidated `Topic` or `Method` page entirely unless refactoring.
- Use `replace` to append new findings or refine existing definitions.
- If a new paper confirms an existing theory, add a citation (e.g., `... also validated in [[Source_Name]]`).

### 3. Conflict Resolution
- If two sources contradict (e.g., one prefers Tanh, another SiLU for the same problem):
  - Document both perspectives.
  - Create a `## Comparative Analysis` or `## Conflicts` section in the relevant `Topic` page.
  - State the context of each study (e.g., "For stiff equations, [[Source_A]] suggests X, while for smooth flows [[Source_B]] suggests Y").

## Page Templates

### Literature Template
- `## Summary`: High-level purpose of the work.
- `## Key Methodology`: Specific algorithms or PINN architectures used.
- `## Key Findings`: Results, accuracy metrics, and unique insights.
- `## Related`: Mandatory links to `[[Topics]]`, `[[Methods]]`, and `[[Systems]]`.

### Topic/Method/System Template
- `## Overview`: Definition and core theory.
- `## Technical Implementation`: How it is applied in the project.
- `## References`: Back-links to `[[Literature]]` sources.

## Formatting Rules
- Use `[[Wikilinks]]` for all internal references.
- Use standard Markdown headers.
- Include LaTeX for math: `\( ... \)` for inline, `\[ ... \]` for blocks.
- Every claim in a `Topic` or `Method` page must cite its source.

## Health Checks (Linting)
Periodically run the following check to identify broken links:
```bash
grep -roE '\[\[[^]]+\]\]' Wiki | sed 's/.*\[\[//;s/\]\]//' | sed -E 's/\|.*//' | sort | uniq > links.txt && find Wiki -name "*.md" | sed 's/.*\///;s/\.md//' | sort | uniq > pages.txt && comm -23 links.txt pages.txt | grep -vE "\.pdf$"
```
- **Action**: Create the missing pages or fix the typos immediately.
