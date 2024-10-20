from typing import List, Dict
from .assistant import Assistant, AssistantResponse
from models import Message, Capability

class CustomModelAssistant(Assistant):
    def __init__(self):
        self.api_url = "https://egovapi.nitec-ai.kz/v1/egov_ai"

    async def send_to_assistant(
        self,
        prompt: str,
        noa_system_prompt: str | None,
        image_bytes: bytes | None,
        message_history: List[Message] | None,
        learned_context: Dict[str, str],
        location_address: str | None,
        local_time: str | None,
        model: str | None,
        web_search: None,
        vision: None,
        speculative_vision: bool
    ) -> AssistantResponse:
        # Make the POST request asynchronously
        payload = {"user_question": prompt}

        # Use an asynchronous HTTP client like `httpx` instead of `requests`
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, json=payload)

        # Handle the response
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data.get("answer", "No answer available.")

            # Create an AssistantResponse object
            returned_response = AssistantResponse(
                response=answer,
                token_usage_by_model={},  # No token usage tracking in this case
                capabilities_used=[Capability.ASSISTANT_KNOWLEDGE],
                debug_tools="",
                timings=""
            )
        else:
            # Handle error case
            returned_response = AssistantResponse(
                response="Error: Failed to get a response from the model.",
                token_usage_by_model={},
                capabilities_used=[Capability.ASSISTANT_KNOWLEDGE],
                debug_tools="",
                timings=""
            )

        return returned_response

Assistant.register(CustomModelAssistant)
