# EANM-AI-QC — Chat Commands

**Purpose**: Reference for chat commands that trigger specific workflows. Type these commands at any point in a chat.

---

## Session Management

### `::new-chat`

**When**: First message in a new chat session.

**What Claude does**:
1. Reads WORKLOG.md → Current State (phase, last item, next item, blockers)
2. Reads WORKLOG.md → Hard Facts (all accumulated knowledge)
3. Reads IMPLEMENTATION_PLAN.md → Session Handoff (status confirmation)
4. Reads REQUIREMENTS.md if any open decision is pending
5. Prints a **status summary**:
   - Current phase and progress
   - Last completed item
   - Hard facts count
   - Any blockers or open questions
6. Lists the **next 3–5 TODO items** with brief descriptions
7. Waits for user approval before starting any work

### `::update`

**When**: End of a session, or after a major implementation milestone.

**What Claude does**:
1. Generates updated **WORKLOG.md** with:
   - Updated Section 1 (Current State)
   - New Session Log entry (what was done, issues, decisions)
   - Any new Hard Facts discovered during the session
   - Updated File Delivery Tracker
2. Generates updated **IMPLEMENTATION_PLAN.md** with:
   - Updated Section 7 (Session Handoff — current status, next item)
   - Any completed items marked as DONE
3. If requirements changed during the session: generates updated **REQUIREMENTS.md** with incremented version
4. Presents all updated files for download
5. Prints a checklist: "Drop these files into the project: [list]"

### `::status`

**When**: Any time during a session to check progress.

**What Claude does**:
1. Prints current phase, item in progress, items completed this session
2. Lists remaining items in current phase
3. Lists any blockers or open questions

### `::hard-fact <description>`

**When**: A non-obvious discovery is made (dependency conflict, API quirk, design decision).

**What Claude does**:
1. Assigns the next HF-NNN number
2. Records the fact with date, context, and impact
3. Confirms: "Recorded as HF-NNN. Will be persisted on next `::update`."

### `::phase-gate`

**When**: All items in a phase appear complete and we want to verify readiness for the next phase.

**What Claude does**:
1. Reads the phase gate checklist from IMPLEMENTATION_PLAN.md
2. Checks each item: DONE / NOT DONE / PARTIAL
3. Prints a pass/fail summary
4. If all pass: "Phase N gate PASSED. Ready for Phase N+1. Proceed?"
5. If any fail: lists what remains

### `::plan <description>`

**When**: Before starting a new implementation step.

**What Claude does**:
1. Describes the implementation step in detail
2. Lists substeps (baby steps)
3. Identifies which source files to read/port
4. Lists any open questions that need answering before starting
5. Waits for user approval: "Approve this plan? (yes / modify / questions)"

### `::next`

**When**: User confirms current substep is complete and wants to proceed.

**What Claude does**:
1. Marks current substep as done
2. Moves to next substep
3. Describes what it will do
4. Waits for confirmation before executing

---

## Quick Reference

| Command | When | Output |
|---|---|---|
| `::new-chat` | Start of new chat | Status + next TODOs |
| `::update` | End of session | Updated WORKLOG + PLAN files |
| `::status` | Anytime | Current progress |
| `::hard-fact <text>` | Discovery made | Records fact |
| `::phase-gate` | End of phase | Gate checklist pass/fail |
| `::plan <text>` | Before starting work | Substep breakdown + approval |
| `::next` | Substep complete | Advance to next substep |
