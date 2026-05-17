---
name: update_wiki
description: Autonomous maintenance, ingestion, structural refactoring, and compounding curation of the PINN-tesi Obsidian Wiki.
---

# update_wiki: Autonomous Knowledge Curation Protocol

## 1. Skill Overview & Philosophy
The `update_wiki` skill transforms Antigravity into an autonomous knowledge curator for the PINN-tesi project. It operates like an expert technical editor maintaining a living ship's logbook ("diario di bordo"). 

Its core philosophy is to maintain a compounding knowledge base where:
- **Atomic Concepts** (Topics/Methods) are pristine, modular, and strictly interconnected via Wikilinks.
- **Physical & Training Details** (PDE residuals, boundary conditions, loss weighting, staged precision, VRAM management) are front-and-center and easily accessible.
- **System Supervision** (System guides like `Viscoelastic_Training.md`) orchestrates the atomic concepts without cluttering them.
- **Structural Hygiene** is actively maintained through periodic consolidation, merging redundant pages, splitting overgrown pages, and ensuring zero broken links via automated LIST linting.

## 2. Invocation & Execution Modes
When the user invokes `update_wiki` (e.g., "Antigravity, esegui update_wiki" or "aggiorna la wiki con quanto fatto oggi"), the agent MUST determine the required mode(s) of operation based on the user's prompt or recent context:
1. **Ingest Mode**: Processing new literature/notes from `Reference/` or newly created project scripts.
2. **Session Sync Mode**: Documenting architectural changes, bug fixes, or VRAM/performance optimizations made during the current pair programming session.
3. **Curator (Refactoring) Mode**: Performing a comprehensive structural overhaul, merging similar concepts, splitting dense pages, and updating the vault hierarchy.

## 3. Step-by-Step Execution Workflow

### Step 1: Context Gathering & State Inspection
- View `PINN-wiki/Wiki/00_Index.md` to understand the current vault hierarchy and existing catalog.
- View `PINN-wiki/Wiki/01_Log.md` to review recent activity.
- Identify the target materials (new files in `Reference/`, recent codebase edits, or specific pages requested for refactoring).

### Step 2: Knowledge Synthesis & Direct Placement
When creating or updating pages, ensure strict adherence to the **Separation of Concerns**:
- **Systems (`Wiki/Systems/`)**: High-level orchestrators, experiment guides, and domain-specific analysis (e.g., `Viscoelastic_Training.md`). Focus on training workflows, phase transitions, and multi-field results.
- **Methods (`Wiki/Methods/`)**: Technical, algorithmic, and software engineering implementations (e.g., `VRAM_Optimization.md`, `Dynamic_Weighting.md`). Focus on exact formulas, PyTorch autograd mechanics, and tensor shapes.
- **Topics (`Wiki/Topics/`)**: Physical and mathematical foundations (e.g., `Viscoelasticity.md`, `Spectral_Bias.md`). Focus on governing equations, constitutive laws, and theoretical derivations.
- **Literature (`Wiki/Literature/`)**: Summaries of external papers/notes from `Reference/`. Focus on methodology, key findings, and back-links to project concepts.

*Rule of Direct Placement*: Never dump raw unstructured text. Always integrate new findings surgically into existing pages using `replace_file_content` or create new atomic pages using `write_to_file`.

### Step 3: Structural Refactoring & Hierarchy Overhaul (Curator Mode)
If the wiki has grown dense or the user requests a structural overhaul:
- **Merge Redundancies**: If two pages cover overlapping concepts (e.g., `Oldroyd-B model.md` and `Viscoelasticity.md`), consolidate the underlying theory into the primary Topic page and convert the secondary page into a redirect or specialized sub-page.
- **Split Overgrown Pages**: If a System or Topic page exceeds 150 lines of dense text, extract specialized sections (e.g., extracting VRAM optimization details into a dedicated Method page) and leave a clean summary with a `[[Wikilink]]`.
- **Hierarchy Alignment**: Ensure all active pages are properly categorized and listed in `Wiki/00_Index.md`.

### Step 4: Activity Logging
- Append a comprehensive, formatted entry to `PINN-wiki/Wiki/01_Log.md` documenting every modified page, new concept, and refactoring decision. Format: `## [YYYY-MM-DD] update_wiki | <Summary of Action>`.

### Step 5: Automated Health Check (LIST Linting) & Self-Healing
Execute the mandatory health check command to verify link integrity across the vault:
```bash
grep -roE '\[\[[^]]+\]\]' PINN-wiki/Wiki | sed 's/.*\[\[//;s/\]\]//' | sed -E 's/\|.*//' | sort | uniq > PINN-wiki/links.txt && find PINN-wiki/Wiki -name "*.md" | sed 's/.*\///;s/\.md//' | sort | uniq > PINN-wiki/pages.txt && comm -23 PINN-wiki/links.txt PINN-wiki/pages.txt | grep -vE "\.pdf$"
```
- **Self-Healing Requirement**: If the command outputs any missing page names (excluding section anchors like `#...`), Antigravity MUST NOT stop. It must immediately use `write_to_file` to create the missing atomic pages (using appropriate templates) or fix typos in the referring pages.
- Clean up temporary files: `rm PINN-wiki/links.txt PINN-wiki/pages.txt`.

### Step 6: Final Summary Report
Conclude the turn by providing the user with a concise, elegant markdown summary of the actions taken, pages created/modified, structural refactoring performed, and confirmation of 100% link integrity.

## 4. Page Templates & Formatting Rules

### Literature Template
```markdown
# [Source Title]

## Summary
High-level purpose and context of the work.

## Key Methodology
Specific PINN architectures, loss balancing, or physical formulations introduced.

## Key Findings & Project Relevance
Core results and direct impact on the PINN-tesi project.

## Related Concepts
- **Topics**: [[Topic_1]], [[Topic_2]]
- **Methods**: [[Method_1]]
- **Systems**: [[System_1]]
```

### Topic / Method / System Template
```markdown
# [Concept Name]

## Overview
Definition, theoretical background, and core mathematical formulation (using LaTeX blocks `$$ ... $$`).

## Technical Implementation & Physical Details
Detailed breakdown of how this concept is implemented in the PINN-tesi codebase (e.g., PyTorch autograd mechanics, loss weighting, boundary conditions, VRAM management).

## References & Back-links
- [[Related_Literature_1]]
- [[Related_System_or_Method]]
```

### Formatting Strictness
- Use `[[Wikilinks]]` for all internal connections.
- Ensure proper LaTeX math formatting (`$math$` inline, `$$math$$` blocks).
- Maintain existing docstrings and comments when interacting with code files.
