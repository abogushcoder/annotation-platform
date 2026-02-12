"""
Validate RAG context injection in JSONL training data export.

Generates JSONL from approved conversations and performs structural validation
to ensure RAG context appears correctly in user messages, follows OpenAI
fine-tuning schema rules, and handles edge cases properly.
"""
import json

from django.core.management.base import BaseCommand

from conversations.models import Conversation, Turn
from conversations.services.export import (
    conversation_to_messages,
    count_tokens,
    generate_jsonl_examples,
    validate_example,
)


class Command(BaseCommand):
    help = 'Validate RAG context in exported JSONL training data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose', action='store_true',
            help='Show detailed output including sample JSONL',
        )
        parser.add_argument(
            '--tag', type=str, default=None,
            help='Only validate conversations with this tag',
        )

    def handle(self, *args, **options):
        verbose = options['verbose']
        tag_filter = options.get('tag')

        self.stdout.write(self.style.MIGRATE_HEADING("RAG Export Validation Report"))
        self.stdout.write("=" * 70)

        results = []  # list of (check_name, passed, detail)

        # ---------------------------------------------------------------
        # Phase 1: Generate examples WITH RAG
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 1: Generate examples with RAG context ---")
        examples_with_rag = generate_jsonl_examples(
            include_rag_context=True,
            tag_filter=tag_filter,
        )
        results.append(self._check(
            "Examples generated (with RAG)",
            len(examples_with_rag) > 0,
            f"{len(examples_with_rag)} examples",
        ))

        # ---------------------------------------------------------------
        # Phase 2: Generate examples WITHOUT RAG
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 2: Generate examples without RAG context ---")
        examples_without_rag = generate_jsonl_examples(
            include_rag_context=False,
            tag_filter=tag_filter,
        )
        results.append(self._check(
            "Examples generated (without RAG)",
            len(examples_without_rag) > 0,
            f"{len(examples_without_rag)} examples",
        ))

        # ---------------------------------------------------------------
        # Phase 3: Structural validation of each example (with RAG)
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 3: Structural validation (with RAG) ---")

        all_valid_json = True
        all_have_messages = True
        all_pass_validate = True
        validation_error_details = []

        for i, ex in enumerate(examples_with_rag):
            # Check 1: Valid JSON (implicitly true since generate_jsonl_examples returns dicts)
            try:
                json.dumps(ex)
            except (TypeError, ValueError) as e:
                all_valid_json = False
                validation_error_details.append(f"  Example {i}: invalid JSON - {e}")

            # Check 2: Has messages
            if 'messages' not in ex or not ex['messages']:
                all_have_messages = False
                validation_error_details.append(f"  Example {i}: missing messages array")

            # Check 3: Passes validate_example()
            errors = validate_example(ex)
            if errors:
                all_pass_validate = False
                validation_error_details.append(f"  Example {i}: {errors}")

        results.append(self._check("All examples are valid JSON", all_valid_json))
        results.append(self._check("All examples have messages array", all_have_messages))
        results.append(self._check(
            "All examples pass validate_example()",
            all_pass_validate,
            "; ".join(validation_error_details) if validation_error_details else None,
        ))

        # ---------------------------------------------------------------
        # Phase 4: RAG context position and format validation
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 4: RAG context position and format checks ---")

        context_in_user_only = True
        context_format_correct = True
        context_has_content = True
        context_followed_by_assistant = True
        total_context_blocks = 0
        examples_with_context = 0
        context_issues = []

        for i, ex in enumerate(examples_with_rag):
            msgs = ex.get('messages', [])
            has_context = False

            for j, msg in enumerate(msgs):
                role = msg.get('role', '')
                content = msg.get('content', '')

                if not content:
                    continue

                if '\n\nContext:\n' in content:
                    total_context_blocks += 1
                    has_context = True

                    # Check: Context only in user messages
                    if role != 'user':
                        context_in_user_only = False
                        context_issues.append(
                            f"  Example {i}, msg {j}: Context block in {role} message"
                        )

                    # Check: Format is "{text}\n\nContext:\n{content}"
                    parts = content.split('\n\nContext:\n', 1)
                    if len(parts) != 2:
                        context_format_correct = False
                        context_issues.append(
                            f"  Example {i}, msg {j}: Bad Context format"
                        )
                    else:
                        user_text, rag_text = parts
                        if not user_text.strip():
                            context_format_correct = False
                            context_issues.append(
                                f"  Example {i}, msg {j}: Empty user text before Context"
                            )
                        if not rag_text.strip():
                            context_has_content = False
                            context_issues.append(
                                f"  Example {i}, msg {j}: Empty RAG content after Context"
                            )

                    # Check: Next message is assistant
                    if j + 1 < len(msgs):
                        next_role = msgs[j + 1].get('role', '')
                        if next_role != 'assistant':
                            context_followed_by_assistant = False
                            context_issues.append(
                                f"  Example {i}, msg {j}: Context not followed by assistant "
                                f"(got {next_role})"
                            )
                    else:
                        context_followed_by_assistant = False
                        context_issues.append(
                            f"  Example {i}, msg {j}: Context is last message (no assistant after)"
                        )

            if has_context:
                examples_with_context += 1

        results.append(self._check(
            "Context blocks only in user messages",
            context_in_user_only,
            "; ".join(context_issues) if not context_in_user_only else None,
        ))
        results.append(self._check(
            "Context format correct ({text}\\n\\nContext:\\n{chunks})",
            context_format_correct,
        ))
        results.append(self._check(
            "Context blocks have non-empty RAG content",
            context_has_content,
        ))
        results.append(self._check(
            "Context blocks followed by assistant message",
            context_followed_by_assistant,
        ))

        # ---------------------------------------------------------------
        # Phase 5: Verify RAG conversations got context injected
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 5: RAG injection coverage ---")

        # Count how many approved conversations have RAG turns
        qs = Conversation.objects.filter(status='approved')
        if tag_filter:
            qs = qs.filter(tags__name=tag_filter)

        convs_with_rag_turns = 0
        for conv in qs.prefetch_related('turns'):
            has_rag = any(
                t.rag_context and not t.is_deleted
                for t in conv.turns.all()
                if t.role == 'agent'
            )
            if has_rag:
                convs_with_rag_turns += 1

        # Check that the number of examples with context matches expectations
        # (not every conv with RAG turns will produce context - failed fetches with
        # empty content won't inject)
        results.append(self._check(
            "Approved conversations with RAG agent turns found",
            convs_with_rag_turns > 0,
            f"{convs_with_rag_turns} conversations",
        ))
        results.append(self._check(
            "Exported examples with Context blocks found",
            examples_with_context > 0,
            f"{examples_with_context} of {len(examples_with_rag)} examples",
        ))

        # ---------------------------------------------------------------
        # Phase 6: No Context blocks when RAG disabled
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 6: RAG-disabled export has no Context blocks ---")

        no_context_without_rag = True
        for i, ex in enumerate(examples_without_rag):
            for msg in ex.get('messages', []):
                content = msg.get('content', '')
                if content and '\n\nContext:\n' in content:
                    no_context_without_rag = False
                    break
            if not no_context_without_rag:
                break

        results.append(self._check(
            "No Context blocks when include_rag_context=False",
            no_context_without_rag,
        ))

        # ---------------------------------------------------------------
        # Phase 7: Tool call validation
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 7: Tool call sequence validation ---")

        tool_sequences_valid = True
        tool_issues = []
        for i, ex in enumerate(examples_with_rag):
            msgs = ex.get('messages', [])
            pending_ids = set()
            for j, msg in enumerate(msgs):
                if 'tool_calls' in msg:
                    if pending_ids:
                        tool_sequences_valid = False
                        tool_issues.append(
                            f"  Example {i}, msg {j}: new tool_calls before previous resolved"
                        )
                    pending_ids = {tc['id'] for tc in msg['tool_calls']}
                elif msg.get('role') == 'tool':
                    tc_id = msg.get('tool_call_id', '')
                    if tc_id in pending_ids:
                        pending_ids.discard(tc_id)
                    else:
                        tool_sequences_valid = False
                        tool_issues.append(
                            f"  Example {i}, msg {j}: orphaned tool response {tc_id}"
                        )
                else:
                    if pending_ids:
                        tool_sequences_valid = False
                        tool_issues.append(
                            f"  Example {i}, msg {j}: unresolved tool_calls before {msg.get('role')}"
                        )
                    pending_ids = set()

        results.append(self._check(
            "Tool call sequences are valid",
            tool_sequences_valid,
            "; ".join(tool_issues) if tool_issues else None,
        ))

        # ---------------------------------------------------------------
        # Phase 8: Deleted turn RAG exclusion
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 8: Deleted turn exclusion ---")

        # Find conversations with deleted RAG turns and verify their content doesn't appear
        deleted_rag_content_leaked = False
        for conv in qs.prefetch_related('turns'):
            deleted_rag_chunks = []
            for t in conv.turns.all():
                if t.is_deleted and t.rag_context and t.role == 'agent':
                    for chunk in t.rag_context:
                        if chunk.get('content'):
                            deleted_rag_chunks.append(chunk['content'][:50])

            if not deleted_rag_chunks:
                continue

            # Check that the exported example for this conv doesn't contain deleted content
            example = conversation_to_messages(conv, include_rag_context=True)
            example_text = json.dumps(example)
            for snippet in deleted_rag_chunks:
                if snippet in example_text:
                    deleted_rag_content_leaked = True
                    break

        results.append(self._check(
            "Deleted turns' RAG context excluded from export",
            not deleted_rag_content_leaked,
        ))

        # ---------------------------------------------------------------
        # Phase 9: System message checks
        # ---------------------------------------------------------------
        self.stdout.write("\n--- Phase 9: System message checks ---")

        system_msg_correct = True
        for i, ex in enumerate(examples_with_rag):
            msgs = ex.get('messages', [])
            if msgs and msgs[0].get('role') == 'system':
                if not msgs[0].get('content', '').strip():
                    system_msg_correct = False

        results.append(self._check(
            "System messages (when present) have non-empty content",
            system_msg_correct,
        ))

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.MIGRATE_HEADING("Summary"))

        passed = sum(1 for _, p, _ in results if p)
        failed = sum(1 for _, p, _ in results if not p)

        for name, p, detail in results:
            status = self.style.SUCCESS("PASS") if p else self.style.ERROR("FAIL")
            line = f"  [{status}] {name}"
            if detail:
                line += f" -- {detail}"
            self.stdout.write(line)

        self.stdout.write(f"\n  {passed} passed, {failed} failed out of {len(results)} checks")

        # Stats
        token_count = count_tokens(examples_with_rag)
        self.stdout.write(f"\n  Total examples:         {len(examples_with_rag)}")
        self.stdout.write(f"  Examples with RAG:      {examples_with_context}")
        self.stdout.write(f"  Total Context blocks:   {total_context_blocks}")
        self.stdout.write(f"  Estimated tokens:       {token_count:,}")

        # ---------------------------------------------------------------
        # Verbose: show sample JSONL
        # ---------------------------------------------------------------
        if verbose and examples_with_rag:
            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.MIGRATE_HEADING("Sample JSONL (first 3 examples)"))
            for i, ex in enumerate(examples_with_rag[:3]):
                self.stdout.write(f"\n--- Example {i + 1} ---")
                self.stdout.write(json.dumps(ex, indent=2, ensure_ascii=False)[:3000])

        if failed:
            self.stderr.write(self.style.ERROR(f"\n{failed} check(s) FAILED."))
        else:
            self.stdout.write(self.style.SUCCESS(f"\nAll {passed} checks passed!"))

    def _check(self, name, passed, detail=None):
        """Record and display a check result."""
        status = self.style.SUCCESS("PASS") if passed else self.style.ERROR("FAIL")
        line = f"  [{status}] {name}"
        if detail:
            line += f" -- {detail}"
        self.stdout.write(line)
        return (name, passed, detail)
