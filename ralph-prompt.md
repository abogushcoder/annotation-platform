You are extensively testing a Fine-Tuning Annotation Platform — a standalone Django web application that was already built.

TARGET DIRECTORY: /home/riba/Desktop/x1-and-prototypes/annotation-platform/
SPEC DOCUMENT: /home/riba/Desktop/x1-and-prototypes/x1-suite-restaurant/docs/fine-tuning-annotation-platform-spec.md

Read the spec document at the start of EVERY iteration to stay aligned with requirements.

TECH STACK:
- Django 6.0.2 + Python 3.13.1 (pyenv)
- PostgreSQL 16 (Docker on port 5434, already running)
- HTMX + Tailwind CSS (CDN, no build step)
- tiktoken for token counting
- Django dev server on port 8000

TEST ACCOUNTS:
- Admin: username=admin, password=admin (role=admin)
- Annotator: username=sarah, password=annotator (role=annotator)

IMPORTANT RULES:
- Work in /home/riba/Desktop/x1-and-prototypes/annotation-platform/
- Activate venv: source .venv/bin/activate
- Run tests: python manage.py test --verbosity=2
- Start dev server: python manage.py runserver (for UI testing)
- Fix ALL issues you find before moving to the next phase
- Commit after each testing phase with a descriptive message
- Do NOT skip the security auditor — it was explicitly excluded by the user

## TESTING PHASES — Complete ALL before outputting the promise

### Phase 1: Comprehensive Unit Testing (test-writer)

Spawn the **test-writer** agent to generate comprehensive tests for the annotation platform. The test-writer should cover:

1. **Model Edge Cases** (conversations/tests.py or new test files):
   - Conversation status transitions: test ALL valid transitions and verify invalid ones are rejected
   - Turn ordering: ensure positions are correct, test bulk creation
   - ToolCall with missing/malformed args (None, empty dict, nested JSON)
   - SystemPrompt: test creating multiple active prompts, ensure only latest is active
   - Agent: test duplicate agent_id handling
   - User role permissions: test all combinations

2. **Export Pipeline Deep Testing**:
   - conversation_to_messages with NO turns (empty conversation)
   - conversation_to_messages with ONLY user turns (no agent)
   - conversation_to_messages with ONLY agent turns (no user)
   - conversation_to_messages with multiple tool calls on same turn
   - conversation_to_messages with edited text vs original (verify edited is used)
   - conversation_to_messages with edited tool call args (verify edited is used)
   - Tool call with missing response_body (null)
   - Tool call with error status code (400, 500)
   - validate_example with malformed tool_calls (missing function key, invalid JSON args)
   - validate_example with duplicate tool_call_ids
   - generate_jsonl_examples with mix of approved and non-approved conversations
   - split_train_validation with edge cases: 0 examples, 1 example, 2 examples
   - count_tokens returns consistent results
   - estimate_training_cost calculation accuracy
   - export_jsonl produces valid JSON on each line
   - Tool definitions: verify all 8 tools have correct schema structure

3. **View/Endpoint Testing**:
   - Authentication: unauthenticated access to all protected routes returns redirect/403
   - Annotator trying to access admin routes returns 403
   - Admin accessing annotator routes works
   - Conversation editor: test with conversation NOT assigned to logged-in user
   - Turn edit: test editing turn on conversation assigned to different user
   - Tool call edit: test with each of the 8 tool types
   - Complete conversation that's not in_progress (should fail)
   - Flag conversation without notes
   - Approve/reject conversation not in completed status
   - Assign conversation already assigned to someone
   - Auto-distribute with no annotators
   - Auto-distribute with uneven distribution
   - Export download with no approved conversations
   - Export download with split option
   - Export download with tool_calls_only filter
   - Export download with agent filter
   - Sync endpoint (mocked ElevenLabs API)
   - Team invite with duplicate username
   - Prompt management: edit, activate, list

4. **ElevenLabs Sync Service Testing**:
   - Mock the ElevenLabs API client
   - Test sync with empty conversation list
   - Test sync with conversations that have no transcript
   - Test sync deduplication (same elevenlabs_id twice)
   - Test sync with pagination (cursor handling)
   - Test transcript parsing with various formats

After generating tests, run them ALL and fix any failures. Iterate until all tests pass.

### Phase 2: UI/UX Review

Spawn the **ui-ux-reviewer** agent to review the following pages. The Django dev server must be running on port 8000. Start it if needed.

Pages to review (both as admin and annotator where applicable):

1. **Login page**: /login/
2. **Annotator dashboard**: /conversations/ (logged in as annotator)
3. **Conversation editor**: /conversations/<id>/ (with test data - turns, tool calls)
4. **Admin dashboard**: /admin-panel/ (logged in as admin)
5. **Agent management**: /admin-panel/agents/
6. **Assignment page**: /admin-panel/assign/
7. **Review queue**: /admin-panel/review/
8. **Review detail**: /admin-panel/review/<id>/
9. **Export page**: /admin-panel/export/
10. **Analytics page**: /admin-panel/analytics/
11. **Team management**: /admin-panel/team/
12. **Prompt management**: /admin-panel/prompts/
13. **403 page**: Test by accessing admin route as annotator
14. **404 page**: Navigate to non-existent URL

For each page, evaluate:
- Visual design consistency (Tailwind styling, spacing, colors)
- Responsive layout (test at different widths)
- Accessibility (contrast, labels, keyboard navigation)
- Loading states and error handling
- Empty states (what does the page look like with no data?)
- Navigation flow and breadcrumbs
- User feedback (success/error messages after actions)

Fix ALL critical and high-priority UI/UX issues found.

### Phase 3: Integration Testing

After unit tests and UI fixes are complete:

1. **Full annotator workflow test**:
   - Log in as annotator
   - View assigned conversations
   - Open conversation editor
   - Verify status auto-transitions to in_progress
   - Edit a turn via HTMX
   - Edit a tool call via HTMX
   - Mark conversation as complete
   - Verify completed conversations appear correctly

2. **Full admin workflow test**:
   - Log in as admin
   - View dashboard stats
   - Assign conversations to annotators
   - Review completed conversations
   - Approve/reject conversations
   - Export approved conversations as JSONL
   - Verify JSONL format is valid for OpenAI fine-tuning
   - Check analytics page reflects current data

3. **JSONL validation**: Verify exported JSONL matches OpenAI's fine-tuning format:
   - Each line is valid JSON
   - Each example has a "messages" array
   - First message is system role
   - Tool calls use correct function calling format
   - Tool responses have matching tool_call_ids
   - Tools array has correct schema structure

Run ALL tests one final time to confirm everything passes.

## FINAL VERIFICATION (must ALL pass before promise)
- [ ] All unit tests pass (python manage.py test --verbosity=2)
- [ ] Test coverage includes model edge cases, export pipeline, views, and sync
- [ ] At least 60+ total tests (up from current 37)
- [ ] UI/UX reviewer has reviewed all key pages
- [ ] Critical UI/UX issues are fixed
- [ ] Full annotator workflow works end-to-end
- [ ] Full admin workflow works end-to-end
- [ ] JSONL export produces valid OpenAI fine-tuning format
- [ ] No uncaught exceptions on any page
- [ ] All fixes are committed to git
