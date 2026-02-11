You are building a Fine-Tuning Annotation Platform — a standalone Django web application.

TARGET DIRECTORY: /home/riba/Desktop/x1-and-prototypes/annotation-platform/
SPEC DOCUMENT: /home/riba/Desktop/x1-and-prototypes/x1-suite-restaurant/docs/fine-tuning-annotation-platform-spec.md

Read the spec document at the start of EVERY iteration to stay aligned.

TECH STACK:
- Django 5.x + Python 3.12 (pyenv)
- PostgreSQL 16 (Docker on port 5434)
- HTMX + Tailwind CSS (CDN, no build step)
- boto3 for S3 audio storage
- tiktoken for token counting

IMPORTANT RULES:
- Work in /home/riba/Desktop/x1-and-prototypes/annotation-platform/
- Use python3 from pyenv (Python 3.13.1)
- Use docker-compose for PostgreSQL on port 5434
- Use Django's built-in dev server on port 8000
- Use Tailwind via CDN link (no npm/node build step)
- Use HTMX via CDN link
- Commit after completing each phase with a descriptive message
- Run Django tests after each phase to verify
- If tests fail, fix them before moving to the next phase

## BUILD PHASES — Complete ALL before outputting the promise

### Phase 1: Foundation (Core CRUD)
- [ ] Create project directory at /home/riba/Desktop/x1-and-prototypes/annotation-platform/
- [ ] Create and activate Python virtual environment (.venv)
- [ ] Install Django 5.x, psycopg2-binary, django-htmx, django-environ, boto3, tiktoken, gunicorn, whitenoise
- [ ] Create docker-compose.yml with PostgreSQL 16 on port 5434
- [ ] Start the PostgreSQL container and verify connection
- [ ] Run django-admin startproject config .
- [ ] Configure settings.py: database (port 5434), installed apps, templates, static files, auth user model
- [ ] Create .env file with DATABASE_URL, SECRET_KEY, DEBUG=True
- [ ] Create 'accounts' app with custom User model (AbstractUser + role field: admin/annotator)
- [ ] Create 'conversations' app with models: Agent, Conversation (7 statuses), Turn, ToolCall, SystemPrompt
- [ ] Run makemigrations and migrate
- [ ] Create superuser (admin@example.com / admin)
- [ ] Implement login/logout views with styled templates
- [ ] Create base.html template with Tailwind CDN, HTMX CDN, navigation bar
- [ ] Verify: Django dev server starts, login works, admin panel accessible
- [ ] Git commit: "Phase 1: Foundation - Django project with models and auth"

### Phase 2: ElevenLabs Sync
- [ ] Create conversations/services/elevenlabs.py — ElevenLabsClient class (list_conversations, get_conversation, get_conversation_audio)
- [ ] Create conversations/services/sync.py — sync logic (fetch conversations, parse transcript into Turn + ToolCall records, handle pagination with cursor)
- [ ] Create conversations/services/audio.py — S3 upload stub (just store audio_s3_key, presigned URL generation placeholder)
- [ ] Create admin_panel app
- [ ] Build Agent management page (admin only): list agents, add agent (agent_id, label, api_key), edit, remove
- [ ] Build Sync page: show agents with last sync time, "Sync Now" button per agent
- [ ] Sync endpoint: POST triggers sync for an agent, creates Conversation + Turn + ToolCall records
- [ ] Handle deduplication: skip conversations already imported (by elevenlabs_id)
- [ ] Store raw_data JSON on Conversation model
- [ ] Verify: Can add an agent and trigger sync (with mock data or real API if key available)
- [ ] Git commit: "Phase 2: ElevenLabs sync - agent management and conversation import"

### Phase 3: Annotator UI
- [ ] Build annotator dashboard: show assigned/in-progress/completed counts, list of conversations
- [ ] Build conversation editor page (core screen): chat-style display with turns
- [ ] Display turns: agent messages right-aligned (blue), customer messages left-aligned (gray)
- [ ] Show tool call cards inline: friendly display with tool_name header, key-value fields, result status
- [ ] Inline turn editing via HTMX: click Edit -> expand textarea with original + editable text, Save/Cancel
- [ ] Tool call editing via HTMX: "Edit Tool Call" opens form with friendly field names per tool_name
- [ ] Implement tool call forms for all 8 tools: create_order, cancel_order, remove_item, modify_item, check_availability, create_reservation, get_specials, get_past_orders
- [ ] Audio player section (placeholder with HTML5 audio element, presigned URL integration)
- [ ] Status transitions: assigned -> in_progress (auto on first edit), in_progress -> completed (button)
- [ ] Flag/skip functionality with notes
- [ ] Annotator notes textarea (saves via HTMX)
- [ ] Conversation list filtering: by status tabs (assigned, in_progress, completed)
- [ ] Verify: Can view conversation, edit turns inline, edit tool calls, mark as completed
- [ ] Git commit: "Phase 3: Annotator UI - conversation editor with HTMX"

### Phase 4: Admin Panel
- [ ] Admin dashboard: overview stats (total, pending, completed, approved counts)
- [ ] Team progress bars: per-annotator completion percentage and throughput
- [ ] Conversation assignment page: filter by agent/status, select conversations, assign to annotator
- [ ] Bulk assignment: select multiple + choose assignee
- [ ] Auto-distribute: evenly distribute unassigned across active annotators
- [ ] Review queue: list completed conversations awaiting review
- [ ] Review page: same editor view + diff highlighting (original vs edited in yellow) + annotator notes
- [ ] Approve/reject buttons: approve sets status=approved, reject sets status=assigned with reviewer_notes
- [ ] Team management: list members, invite (create account with temp password), deactivate
- [ ] System prompt management: list prompts, create new version, set active, edit
- [ ] Verify: Can assign conversations, review, approve/reject, manage team
- [ ] Git commit: "Phase 4: Admin panel - assignment, review, team management"

### Phase 5: Export Pipeline
- [ ] Create conversations/services/export.py — JSONL export service
- [ ] Turn -> OpenAI message transformation (user->user, agent->assistant, tool calls->function format)
- [ ] System prompt injection as first message in every example
- [ ] Tool call -> function calling format: assistant message with tool_calls array + tool message with response
- [ ] Include tools array with full schema for all 8 tools
- [ ] Validation: every example has user + assistant messages, valid JSON args, unique tool_call_ids, no empty content
- [ ] Train/validation split option (80/20 random)
- [ ] Token counting with tiktoken (estimate training cost)
- [ ] Export page UI: filter options (all approved, by agent, by date range), checkboxes for options, preview, download
- [ ] Preview endpoint: show first 3 examples formatted
- [ ] Download endpoint: generate and serve .jsonl file(s)
- [ ] Verify: Can export approved conversations as valid JSONL, token count displays
- [ ] Git commit: "Phase 5: Export pipeline - JSONL generation with validation"

### Phase 6: Analytics and Polish
- [ ] Per-annotator metrics: completion rate, avg time per conversation, conversations/hour
- [ ] Edit rate and tool call edit rate tracking
- [ ] Rejection rate and flag rate
- [ ] Analytics dashboard page with charts/progress bars
- [ ] Pipeline visualization: Unassigned -> Assigned -> In Progress -> Completed -> Approved funnel
- [ ] Export history log (when, how many, by whom)
- [ ] UI polish: consistent styling, responsive layout, loading states, error messages
- [ ] Role-based navigation: annotators see their dashboard, admins see admin panel
- [ ] Middleware: restrict admin routes to admin role
- [ ] 404 and permission denied pages
- [ ] Verify: Analytics page shows metrics, navigation is role-based, all pages are polished
- [ ] Git commit: "Phase 6: Analytics, polish, and role-based access"

## FINAL VERIFICATION (must ALL pass before promise)
- [ ] PostgreSQL container running on port 5434
- [ ] Django dev server starts without errors on port 8000
- [ ] Can log in as admin and annotator
- [ ] Can add an agent and see agent list
- [ ] Conversation models are properly migrated
- [ ] Annotator dashboard shows assigned conversations
- [ ] Conversation editor displays turns in chat style
- [ ] Turn editing works via HTMX (inline edit, save, cancel)
- [ ] Tool call editing works with per-tool forms
- [ ] Admin can assign conversations to annotators
- [ ] Admin can review and approve/reject conversations
- [ ] Export produces valid JSONL with system prompt and tool schemas
- [ ] Analytics page displays metrics
- [ ] All Django tests pass (python manage.py test)
- [ ] No uncaught exceptions on any page
