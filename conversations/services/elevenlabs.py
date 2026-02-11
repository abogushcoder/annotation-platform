import requests


class ElevenLabsClient:
    BASE_URL = "https://api.elevenlabs.io/v1/convai"

    def __init__(self, api_key: str):
        self.headers = {"xi-api-key": api_key}

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

    def get_conversation_audio(self, conversation_id: str) -> bytes:
        resp = requests.get(
            f"{self.BASE_URL}/conversations/{conversation_id}/audio",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.content
