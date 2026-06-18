---
name: handoff
description: Write or update a handoff document summarizing the current conversation and context so the next agent can continue the work effectively.
---

# Handoff Document Generation

When invoked, your task is to create or update a handoff document that summarizes the current state of the conversation and the project. This allows a fresh agent to seamlessly pick up where you left off.

## Process
1. **Locate or Create**: Check if `HANDOFF.md` already exists in the root of the project.
   - If it exists, read it first using your file viewing tools to understand prior context before updating.
   - If it does not exist, you will create a new one.
2. **Compile the Content**: The document must be structured with the following sections:
   - **Goal**: What the project or current task is trying to accomplish.
   - **Current Progress**: What has been completed so far in the current session.
   - **What Worked**: Approaches, designs, or implementations that succeeded.
   - **What Didn't Work**: Approaches that failed (this is crucial so the next agent doesn't repeat the same mistakes).
   - **Next Steps**: Clear, actionable items for the next session.
   - **Suggested Skills**: Recommend any specific skills the next agent should invoke or be aware of to aid in their tasks.
3. **Handle Arguments (Optional)**: If the user passed arguments or additional context when invoking this skill, treat them as a description of what the next session will focus on and tailor the document accordingly (especially the "Next Steps").
4. **Avoid Duplication**: Do not duplicate content already captured in other artifacts (such as PRDs, implementation plans, ADRs, issues, commits, or diffs). Instead, reference them by their absolute file paths or URLs.
5. **Redact Sensitive Info**: Strictly ensure that any sensitive information, such as API keys, passwords, or personally identifiable information, is redacted from the handoff document.
6. **Save**: Save the final content to `HANDOFF.md` in the project root.
7. **Notify User**: Tell the user the absolute file path of the saved `HANDOFF.md` so they can start a fresh conversation and provide that document to the new agent.
