import requests


class ElevenLabsClient:
    BASE_URL = "https://api.elevenlabs.io/v1/convai"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"xi-api-key": api_key}

    def list_agents(self) -> list[dict]:
        """Fetch all agents from the ElevenLabs account."""
        resp = requests.get(
            f"{self.BASE_URL}/agents",
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json().get("agents", [])

    def get_agent(self, agent_id: str) -> dict:
        """Fetch full agent config including system prompt."""
        resp = requests.get(
            f"{self.BASE_URL}/agents/{agent_id}",
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json()

    def list_conversations(self, agent_id: str, page_size: int = 100,
                           cursor: str | None = None) -> dict:
        params = {"agent_id": agent_id, "page_size": page_size}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(
            f"{self.BASE_URL}/conversations",
            headers=self.headers, params=params
        )
        resp.raise_for_status()
        return resp.json()

    def get_conversation(self, conversation_id: str) -> dict:
        resp = requests.get(
            f"{self.BASE_URL}/conversations/{conversation_id}",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()

    def get_conversation_audio(self, conversation_id: str, timeout: int = 30) -> bytes:
        resp = requests.get(
            f"{self.BASE_URL}/conversations/{conversation_id}/audio",
            headers=self.headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content

    def get_kb_chunk(self, documentation_id: str, chunk_id: str) -> dict:
        """Fetch a single knowledge base chunk's content.

        Returns dict with 'content' key containing the chunk text.
        Endpoint: GET /v1/convai/knowledge-base/{doc_id}/chunk/{chunk_id}
        """
        resp = requests.get(
            f"{self.BASE_URL}/knowledge-base/{documentation_id}/chunk/{chunk_id}",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
