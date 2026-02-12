import logging

from django.core.management.base import BaseCommand

from conversations.models import Conversation, Turn
from conversations.services.elevenlabs import ElevenLabsClient

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Backfill RAG context for existing conversations from raw_data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be updated without making changes',
        )
        parser.add_argument(
            '--agent-id', type=int,
            help='Only backfill conversations for a specific agent (DB pk)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        agent_id = options.get('agent_id')

        qs = Conversation.objects.exclude(raw_data={}).exclude(raw_data__isnull=True)
        if agent_id:
            qs = qs.filter(agent_id=agent_id)

        conversations = qs.select_related('agent')
        total = conversations.count()
        self.stdout.write(f"Scanning {total} conversations with raw_data...")

        updated_turns = 0
        skipped = 0
        errors = 0

        for conv in conversations.iterator():
            transcript = conv.raw_data.get('transcript', [])
            if not transcript:
                continue

            # Check if any turn in the transcript has rag_retrieval_info
            has_rag = any(t.get('rag_retrieval_info') for t in transcript)
            if not has_rag:
                continue

            client = ElevenLabsClient(conv.agent.elevenlabs_api_key)
            turns_by_position = {t.position: t for t in conv.turns.all()}

            for position, turn_data in enumerate(transcript):
                rag_info = turn_data.get('rag_retrieval_info')
                if not rag_info:
                    continue

                turn = turns_by_position.get(position)
                if not turn:
                    continue

                # Skip turns that already have rag_context
                if turn.rag_context:
                    skipped += 1
                    continue

                chunks_meta = rag_info.get('chunks', [])
                rag_chunks = []
                for chunk_meta in chunks_meta:
                    doc_id = chunk_meta.get('document_id', '')
                    chunk_id = chunk_meta.get('chunk_id', '')
                    distance = chunk_meta.get('vector_distance')
                    if doc_id and chunk_id:
                        try:
                            chunk_data = client.get_kb_chunk(doc_id, chunk_id)
                            rag_chunks.append({
                                'document_id': doc_id,
                                'chunk_id': chunk_id,
                                'content': chunk_data.get('content', ''),
                                'vector_distance': distance,
                            })
                        except Exception as e:
                            logger.warning(f"Failed to fetch KB chunk {doc_id}/{chunk_id}: {e}")
                            rag_chunks.append({
                                'document_id': doc_id,
                                'chunk_id': chunk_id,
                                'content': '',
                                'vector_distance': distance,
                                'fetch_error': str(e),
                            })
                            errors += 1

                if rag_chunks:
                    if dry_run:
                        self.stdout.write(
                            f"  [DRY RUN] Would update turn {turn.pk} "
                            f"(conv {conv.elevenlabs_id}, pos {position}) "
                            f"with {len(rag_chunks)} RAG chunks"
                        )
                    else:
                        turn.rag_context = rag_chunks
                        turn.save(update_fields=['rag_context'])
                    updated_turns += 1

        self.stdout.write(self.style.SUCCESS(
            f"Done. {'[DRY RUN] ' if dry_run else ''}"
            f"Updated: {updated_turns}, Skipped (already filled): {skipped}, "
            f"Chunk fetch errors: {errors}"
        ))
