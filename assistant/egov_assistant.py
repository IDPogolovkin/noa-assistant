import requests
from typing import List, Dict, Optional
from .assistant import Assistant, AssistantResponse
from models import Message, Capability, Role
from web_search import WebSearch
from vision import Vision

class CustomModelAssistant(Assistant):
    def __init__(self):
        self.api_url = "https://egovapi.nitec-ai.kz/v1/egov_ai"

    async def send_to_assistant(
        self,
        prompt: str,
        noa_system_prompt: Optional[str],
        image_bytes: Optional[bytes],
        message_history: Optional[List[Message]],
        learned_context: Dict[str, str],
        location_address: Optional[str],
        local_time: Optional[str],
        model: Optional[str],
        web_search: Optional[WebSearch],
        vision: Optional[Vision],
        speculative_vision: bool
    ) -> AssistantResponse:
        print(f"Assistant received prompt: {prompt}")
        # Make the POST request
        if not prompt.strip():
            # Handle empty prompt case
            return AssistantResponse(
                response="Error: Prompt cannot be empty.",
                token_usage_by_model={},
                capabilities_used=[Capability.ASSISTANT_KNOWLEDGE],
                debug_tools={},
                timings={},
                image=None,
                topic_changed=False,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0
            )
        
        # Determine if the topic has changed
        topic_changed = self.detect_topic_change(prompt, message_history)

        payload = {"user_question": prompt}
        print(f"Payload for egov API: {payload}")
        response = requests.post(self.api_url, json=payload)
        print(f"Response from eGov API: {response.status_code} {response.text}")

        # Handle the response
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data.get("answer", "No answer available.")

            # Create an AssistantResponse object
            returned_response = AssistantResponse(
                response=answer,
                token_usage_by_model={},  # No token usage tracking in this case
                capabilities_used=[Capability.ASSISTANT_KNOWLEDGE],
                debug_tools={},  # Changed from "" to {}
                timings={},      # Changed from "" to {}
                image=None,      # Explicitly set image to None
                topic_changed=topic_changed,
                total_tokens=0,  # Added default values
                input_tokens=0,  # Added default values
                output_tokens=0  # Added default values
            )

        else:
            # Handle error case
            returned_response = AssistantResponse(
                response="Error: Failed to get a response from the model.",
                token_usage_by_model={},
                capabilities_used=[Capability.ASSISTANT_KNOWLEDGE],
                debug_tools={},
                timings={},
                image=None,  # Explicitly set image to None
                topic_changed=False,
                total_tokens=0,  # Added default values
                input_tokens=0,  # Added default values
                output_tokens=0  # Added default values
            )

        return returned_response
    
    def detect_topic_change(self, prompt: str, message_history: Optional[List[Message]]) -> bool:
        if not message_history or len(message_history) == 0:
            return True

        last_user_message = None
        for message in reversed(message_history):
            if message.role == Role.USER:
                last_user_message = message.content
                break

        if last_user_message:
            return prompt.strip() != last_user_message.strip()
        else:
            return True
    
Assistant.register(CustomModelAssistant)
