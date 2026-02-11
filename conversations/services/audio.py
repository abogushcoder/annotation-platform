import logging

logger = logging.getLogger(__name__)


def upload_audio_to_s3(agent_id: str, conversation_id: str, audio_bytes: bytes,
                        track: str = 'full') -> str:
    """Upload conversation audio to S3. Returns the S3 key.

    Stub implementation — stores the key pattern but doesn't actually upload.
    """
    s3_key = f"audio/{agent_id}/{conversation_id}/{track}.mp3"
    logger.info(f"S3 upload stub: would upload to {s3_key} ({len(audio_bytes)} bytes)")
    return s3_key


def get_presigned_url(s3_key: str, expires_in: int = 3600) -> str | None:
    """Generate a presigned URL for an S3 audio file.

    Stub implementation — returns None since S3 is not configured.
    """
    if not s3_key:
        return None
    logger.info(f"Presigned URL stub: would generate URL for {s3_key}")
    return None
