# Annotation Platform - User Guide

A fine-tuning data pipeline that takes raw phone conversations from ElevenLabs voice AI agents, lets humans review and correct them, then exports them as JSONL training data for OpenAI fine-tuning.

```
ElevenLabs conversations → Import → Annotate → Review → Export → Fine-tune
```

---

## Table of Contents

- [Admin Guide](#admin-guide)
  - [Dashboard](#1-dashboard)
  - [Agent Management](#2-agent-management)
  - [Assignment](#3-assignment)
  - [System Prompts](#4-system-prompts)
  - [Team Management](#5-team-management)
  - [Review Queue](#6-review-queue)
  - [Export](#7-export)
  - [Analytics](#8-analytics)
- [Annotator Guide](#annotator-guide)
  - [Conversation List](#1-login--conversation-list)
  - [Opening a Conversation](#2-opening-a-conversation)
  - [Editing Turns](#3-editing-turns)
  - [Deleting / Restoring Turns](#4-deleting--restoring-turns)
  - [Training Weight Toggle](#5-training-weight-toggle)
  - [Editing Tool Calls](#6-editing-tool-calls)
  - [Inserting New Turns](#7-inserting-new-turns)
  - [RAG Context Display](#8-rag-context-display)
  - [Tags and Notes](#9-tags-and-notes)
  - [Completing or Flagging](#10-completing-or-flagging)
- [End-to-End Workflow](#typical-end-to-end-workflow)
- [URL Reference](#key-urls-reference)

---

## Admin Guide

### 1. Dashboard

Navigate to `http://localhost:8000/login/` and log in with admin credentials. You're redirected to the **Admin Dashboard** (`/admin-panel/`) which shows:

- **Stats cards**: Total conversations, Unassigned, Assigned, In Progress, Completed, Approved, Flagged
- **Per-annotator table**: Each annotator's assigned count, completed count, and completion rate
- **Review queue preview**: The 10 most recently completed conversations awaiting your review

### 2. Agent Management

**URL:** `/admin-panel/agents/`

This is where you connect your ElevenLabs voice agents.

**Adding an agent:**

1. Click **"+ Add Agent"**
2. The page calls the ElevenLabs API (using the key in `.env`) and populates a dropdown with all agents in your workspace
3. Select an agent — the platform auto-fills its name and agent ID
4. Save — the agent's **system prompt is automatically imported** from the ElevenLabs config and saved as a SystemPrompt record

**Syncing conversations:**

1. On the agents list, click **"Sync"** next to an agent
2. The platform calls the ElevenLabs API, paginates through all conversations for that agent, and imports new ones
3. For each conversation it imports: the full transcript (turns), all tool calls with arguments and responses, and RAG knowledge base chunks (fetched individually from the KB API)
4. After sync, the page shows "N imported, N skipped, N errors"
5. Sync is incremental — already-imported conversations (matched by `elevenlabs_id`) are skipped

> **Note:** Syncing can be slow if there are many conversations with RAG chunks, since each chunk requires a separate API call.

### 3. Assignment

**URL:** `/admin-panel/assign/`

After sync, conversations land as **"Unassigned"**. You need to assign them to annotators.

**Manual assignment:**

1. Use the tabs to filter: Unassigned / Assigned / In Progress / Completed / Flagged
2. Check the boxes next to conversations you want to assign
3. Select an annotator from the dropdown at the top
4. Click **"Assign"** — those conversations move to "Assigned" status

**Auto-distribute:**

Click **"Auto Distribute"** to round-robin all unassigned conversations evenly across active annotators.

You can also filter by agent or search by conversation content/ID.

### 4. System Prompts

**URL:** `/admin-panel/prompts/`

Manage the system prompt that gets included in exported JSONL.

- **Auto-imported prompts** appear when you add a new agent (extracted from ElevenLabs agent config)
- Only **one prompt can be active** at a time — activating one deactivates all others
- The active prompt is used as the `"system"` message in exported training examples
- You can create custom prompts or edit existing ones

### 5. Team Management

**URL:** `/admin-panel/team/`

- **Invite annotators**: Set username, password, and role (annotator)
- **Toggle active/inactive**: Deactivated users can't log in
- View all team members and their roles

### 6. Review Queue

**URL:** `/admin-panel/review/`

When an annotator marks a conversation as "Completed", it appears here.

**Reviewing a conversation:**

1. Click on a conversation to see the full annotated version
2. The **Edit Summary Bar** at the top shows counts:
   - Turns edited
   - Turns deleted
   - Turns inserted
   - Tool calls deleted
   - Weight overrides
   - Turns with RAG context
3. You can see all changes visually:
   - **Edited turns**: Original text shown with strikethrough, edited text below
   - **Deleted turns**: Shown in red with reduced opacity
   - **Inserted turns**: Green dashed border with "Inserted" badge
   - **Tool call edits**: "Args Edited" badge on modified tool calls
   - **Weight overrides**: Visual indicator showing W:0 (red) or W:1 (green)
   - **Annotator notes**: Displayed at the bottom
4. **Approve** — conversation moves to "Approved" (eligible for export)
5. **Reject** — conversation reverts to "Assigned" and goes back to the annotator's queue. You can add reviewer notes explaining what needs to be fixed.

**Bulk approve:** Select multiple conversations and approve them all at once.

### 7. Export

**URL:** `/admin-panel/export/`

Export approved conversations as JSONL for OpenAI fine-tuning.

**Configuration options:**

| Option | Description |
|--------|-------------|
| Filter by agent | Export only conversations from a specific agent |
| Filter by tag | Export only conversations with a specific tag |
| Tool calls only | Only include conversations that have tool calls |
| Include system prompt | Add the active system prompt as the first message |
| Include tools | Include the `tools` array (function definitions) in each example |
| Include RAG context | Inject knowledge base chunks into user messages as `\n\nContext:\n...` blocks |

**Preview:** Click **"Preview First 3 Examples"** to see syntax-highlighted JSONL before downloading.

**Token & cost estimate:** The page shows estimated token count (via `tiktoken`) and fine-tuning cost based on OpenAI pricing ($25/1M tokens, 3 epochs).

**Download formats:**

- **Single JSONL file** — all examples in one file
- **Train/Val ZIP** — 80/20 split into `train.jsonl` and `val.jsonl`

Each export is logged in an audit table (who exported, how many conversations, token count).

**Minimum requirement:** At least 10 approved conversations are needed before export is enabled.

### 8. Analytics

**URL:** `/admin-panel/analytics/`

- **Pipeline funnel**: Visual breakdown of conversations at each status
- **Per-annotator metrics**: Completion rate, rejection rate, flag rate, turn edit rate, tool call edit rate
- **Export history**: Last 10 exports with metadata
- **Overall edit rates** across all annotators

---

## Annotator Guide

### 1. Login & Conversation List

Navigate to `http://localhost:8000/login/`, log in with your annotator credentials. You're redirected to `/conversations/` — your personal workspace.

**Tabs:**

- **Assigned** — Conversations assigned to you, not yet started
- **In Progress** — Conversations you've opened and are working on
- **Completed** — Conversations you've finished

Each card shows: agent name, turn count, call duration, timestamp.

### 2. Opening a Conversation

Click on a conversation card to open the editor. **Opening an "Assigned" conversation automatically transitions it to "In Progress".**

The editor shows:

- **Header**: Conversation ID, agent, duration, status
- **Audio player** (if the conversation has audio): Play back the actual phone call
- **Chat transcript**: Customer messages on the left, Agent messages on the right, with timestamps

### 3. Editing Turns

Click the **"Edit"** button on any turn to open an inline edit form.

- Modify the text (fix transcription errors, improve phrasing, etc.)
- Save — the turn shows an **"Edited" badge**
- The original text appears with strikethrough above the edited version
- This is useful for correcting speech-to-text errors or improving agent responses for training

### 4. Deleting / Restoring Turns

Click **"Delete"** on a turn to soft-delete it.

- The turn appears dimmed with red strikethrough and a **"Deleted" badge**
- A **"Restore"** button appears to undo the deletion
- Deleted turns are **excluded from the exported JSONL** but remain visible in the editor

**Use this for:** empty turns ("..."), irrelevant small talk, duplicate turns, transcription artifacts.

### 5. Training Weight Toggle

Click the **weight button** on agent turns to cycle through:

| State | Visual | Meaning |
|-------|--------|---------|
| Auto (null) | Gray | Platform decides (auto-assigns W:0 to agent turns before first user turn) |
| W: 1 | Green | Force the model to learn from this turn |
| W: 0 | Red | Force the model to NOT learn from this turn |

**Use cases:**

- Set W:0 on generic greetings you don't want the model to memorize
- Set W:1 on complex, high-quality responses you want emphasized
- Leave as Auto for most turns

### 6. Editing Tool Calls

If a turn contains tool calls (e.g., `create_order`, `check_availability`), they appear as expandable cards below the turn.

Click **"Edit"** on a tool call to modify its **arguments**. For example:

- Fix a customer name: `"Alex"` -> `"Alexander"`
- Correct an item: `"large pepperoni"` -> `"medium pepperoni"`

Edited tool calls get an **"Args Edited"** badge. The original arguments are preserved.

You can also **delete** tool calls (soft-delete, same as turns).

### 7. Inserting New Turns

Click the **"+"** button between any two turns to insert a new one.

1. Select the **role** (Customer or Agent)
2. Type the text
3. Save — the turn appears with a green dashed border and **"Inserted"** badge

**Use cases:**

- Fill in transcription gaps where speech wasn't captured
- Add a missing customer clarification
- Insert a synthetic training example

### 8. RAG Context Display

Some agent turns show a purple collapsible **"RAG Context (N chunks)"** section. Expand it to see:

- The knowledge base chunks the AI retrieved to answer the customer's question
- Each chunk shows its content and **vector distance** (lower = more relevant)
- This helps you understand why the agent gave a particular response

You don't need to edit RAG context — it's automatically included in exports if the admin enables the "Include RAG context" option.

### 9. Tags and Notes

**Tags:** Add categorical labels to conversations (e.g., "complex-order", "off-topic", "great-example"). Tags can be used by admins to filter exports.

**Notes:** A text area at the bottom of the editor for your observations. Notes are visible to the admin during review. Auto-saves on change.

### 10. Completing or Flagging

When you're done annotating:

- **"Mark as Completed"** — sends the conversation to the admin's review queue
- **"Flag for Review"** — marks it as problematic with a note explaining the issue (e.g., "Conversation is mostly unintelligible", "Agent made a serious error")

After completing, the conversation becomes **read-only** in your view.

---

## Conversation Status Flow

```
unassigned ──> assigned ──> in_progress ──> completed ──> approved
                  ^                            │
                  │                            v
                  └──────── rejected ──────── flagged
```

| Status | Who sets it | Meaning |
|--------|-------------|---------|
| Unassigned | System (on sync) | Newly imported, not assigned to anyone |
| Assigned | Admin | Given to an annotator, not yet opened |
| In Progress | Auto (on open) | Annotator has opened and is working on it |
| Completed | Annotator | Annotation finished, waiting for admin review |
| Flagged | Annotator | Problematic conversation, needs admin attention |
| Approved | Admin | Passed review, eligible for JSONL export |
| Rejected | Admin | Failed review, sent back to annotator for rework |

---

## Typical End-to-End Workflow

```
1. Admin adds ElevenLabs agent           /admin-panel/agents/add/
2. Admin syncs conversations             /admin-panel/agents/ (Sync button)
3. Admin assigns to annotators           /admin-panel/assign/
4. Annotator opens & edits conversations /conversations/<id>/
5. Annotator marks as complete           "Mark as Completed" button
6. Admin reviews & approves/rejects      /admin-panel/review/<id>/
7. Admin exports approved as JSONL       /admin-panel/export/
8. JSONL uploaded to OpenAI fine-tuning  (external)
```

---

## JSONL Export Format

Each exported conversation becomes one training example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a restaurant phone ordering assistant..."},
    {"role": "user", "content": "Hi, I'd like to order a pizza"},
    {"role": "assistant", "content": "Sure! What kind of pizza?", "weight": 1},
    {"role": "user", "content": "A large pepperoni\n\nContext:\nPIZZA MENU\nLarge Pepperoni - $14.99\nMedium Margherita - $9.99"},
    {"role": "assistant", "content": "Great choice!", "weight": 1, "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "create_order", "arguments": "{...}"}}]},
    {"role": "tool", "tool_call_id": "call_001", "content": "{\"success\": true}"}
  ],
  "tools": [...],
  "parallel_tool_calls": false
}
```

Key behaviors:

- **Deleted turns/tool calls** are excluded
- **Edited text/args** are used instead of originals when available
- **Weight** is auto-assigned: agent turns before the first user turn get `weight: 0`
- **RAG context** (when enabled) is injected into the user message preceding the agent turn that used it, as a `\n\nContext:\n...` block
- **Tool call IDs** are sequentially generated (`call_001`, `call_002`, etc.)

---

## Key URLs Reference

| URL | Role | Purpose |
|-----|------|---------|
| `/login/` | Both | Login |
| `/conversations/` | Annotator | Conversation list |
| `/conversations/<id>/` | Annotator | Conversation editor |
| `/admin-panel/` | Admin | Dashboard |
| `/admin-panel/agents/` | Admin | Agent management |
| `/admin-panel/assign/` | Admin | Assignment |
| `/admin-panel/review/` | Admin | Review queue |
| `/admin-panel/export/` | Admin | JSONL export |
| `/admin-panel/analytics/` | Admin | Analytics |
| `/admin-panel/team/` | Admin | Team management |
| `/admin-panel/prompts/` | Admin | System prompts |

---

## Management Commands

```bash
# Create an admin user
python manage.py setup_admin

# Seed test data (10 conversations covering all RAG scenarios)
python manage.py seed_rag_test_data

# Backfill RAG context for previously imported conversations
python manage.py backfill_rag_context [--dry-run] [--agent-id ID]

# Validate RAG export correctness (9 validation phases)
python manage.py validate_rag_export [--verbose] [--tag TAG]
```

---

## Environment Setup

Required in `.env`:

```env
# ElevenLabs API key (required for agent sync)
ELEVENLABS_API_KEY=sk_...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5434/annotation_db
```

Start services:

```bash
docker-compose up -d    # PostgreSQL
python manage.py migrate
python manage.py setup_admin
python manage.py runserver
```
