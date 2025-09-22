"""Intelligent LangGraph agent service for voice conversations."""

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import tempfile
import logging

from services.asr_service import asr_service
from services.tts_service import get_tts_service
from components.ports import SynthesisRequest
from services.llm_service import llm_service

logger = logging.getLogger(__name__)


@tool
def transcribe_audio(audio_data: str, filename: str = "audio.wav") -> str:
    """Transcribe audio data to text using ASR service."""
    try:
        logger.info(f"Tool: Transcribing audio file: {filename}")

        # Decode base64 audio data
        import base64
        audio_bytes = base64.b64decode(audio_data)

        # Use asyncio.run to properly handle the async call
        import asyncio

        async def run_transcription():
            return await asr_service.transcribe_audio(
                audio_file=audio_bytes,
                filename=filename,
                model_name="base"
            )

        # Run the async function properly
        result = asyncio.run(run_transcription())

        transcript = result.get("text", "") if isinstance(result, dict) else str(result)
        logger.info(f"Tool: Transcription successful: {transcript[:50]}...")
        return f"TRANSCRIPT: {transcript}"

    except Exception as e:
        error_msg = f"Error transcribing audio: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@tool
def process_with_llm(text: str) -> str:
    """Process text through LLM to generate a response."""
    try:
        logger.info(f"Tool: Processing with LLM: {text[:50]}...")

        #response = f"I understand you said: '{text}'. This is a placeholder response from the AI Research Assistant."
        response= llm_service.model_inference(text)

        logger.info(f"Tool: LLM response: {response[:50]}...")
        return f"LLM_RESPONSE: {response}"

    except Exception as e:
        error_msg = f"Error processing with LLM: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@tool
def synthesize_speech(text: str) -> str:
    """Convert text to speech using TTS service and return audio file path."""
    try:
        logger.info(f"Tool: Synthesizing speech for: {text[:50]}...")

        # Use existing TTS service
        tts_engine = get_tts_service()

        # Create synthesis request
        request = SynthesisRequest(
            text=text,
            voice_id="en_female_1",
            sample_rate=22050
        )

        # Synthesize audio
        result = tts_engine.synthesize_sync(request)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_file.name, "wb") as f:
            f.write(result.audio)

        audio_path = temp_file.name
        logger.info(f"Tool: Audio synthesized to: {audio_path}")
        return f"AUDIO_FILE: {audio_path}"

    except Exception as e:
        error_msg = f"Error synthesizing speech: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


class IntelligentVoiceAgent:
    def __init__(self):
        self.tools = [transcribe_audio, process_with_llm, synthesize_speech]
        self.tool_node = ToolNode(self.tools)
        self.memory = MemorySaver()
        self.graph = self._create_graph()
        logger.info("Intelligent Voice Agent initialized")

    def _create_graph(self):
        workflow = StateGraph(MessagesState)

        workflow.add_node("coordinator", self._coordinate_tools)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "coordinator")

        return workflow.compile(checkpointer=self.memory)

    def _coordinate_tools(self, state: MessagesState):
        """Intelligent coordinator that decides which tools to call."""
        messages = state["messages"]
        last_message = messages[-1]

        print(f"🔄 COORDINATOR: Processing message type: {type(last_message).__name__}")
        if hasattr(last_message, 'content') and last_message.content:
            print(f"🔄 COORDINATOR: Content: {last_message.content[:100]}...")

        logger.info(f"Coordinator: Processing message type: {type(last_message).__name__}")
        if hasattr(last_message, 'content') and last_message.content:
            logger.info(f"Coordinator: Last message content: {last_message.content[:100]}...")

        # Check if this is the initial audio processing request
        if isinstance(last_message, HumanMessage):
            content = last_message.content
            if "AUDIO_DATA:" in content:
                # Extract audio data and filename from content
                parts = content.split("AUDIO_DATA:")
                if len(parts) > 1:
                    audio_part = parts[1]
                    if "|" in audio_part:
                        audio_data, filename = audio_part.split("|", 1)
                    else:
                        audio_data, filename = audio_part, "audio.wav"

                    logger.info(f"Coordinator: Found audio data, starting transcription for {filename}")
                    tool_call = {
                        "name": "transcribe_audio",
                        "args": {"audio_data": audio_data, "filename": filename},
                        "id": "transcribe_1"
                    }
                    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        # Check for tool results and decide next step
        # Look for the most recent tool result
        tool_result_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and any(msg.content.startswith(prefix) for prefix in ["TRANSCRIPT:", "LLM_RESPONSE:", "AUDIO_FILE:", "ERROR:"]):
                tool_result_msg = msg
                break

        if tool_result_msg:
            content = tool_result_msg.content
            logger.info(f"Coordinator: Found tool result: {content[:50]}...")

            # If we just got a transcription, process with LLM
            if content.startswith("TRANSCRIPT:"):
                transcript = content.replace("TRANSCRIPT: ", "")
                tool_call = {
                    "name": "process_with_llm",
                    "args": {"text": transcript},
                    "id": "llm_1"
                }
                logger.info("Coordinator: Starting LLM processing")
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

            # If we got LLM response, synthesize speech
            elif content.startswith("LLM_RESPONSE:"):
                response_text = content.replace("LLM_RESPONSE: ", "")
                tool_call = {
                    "name": "synthesize_speech",
                    "args": {"text": response_text},
                    "id": "tts_1"
                }
                logger.info("Coordinator: Starting speech synthesis")
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

            # If we have audio file, we're done
            elif content.startswith("AUDIO_FILE:"):
                logger.info("Coordinator: Process complete - audio file generated")
                return {"messages": [AIMessage(content="Voice processing complete")]}

            # If there's an error, stop processing
            elif content.startswith("ERROR:"):
                logger.error(f"Coordinator: Tool error encountered: {content}")
                return {"messages": [AIMessage(content="Error in processing")]}

        # No more tools needed
        logger.info("Coordinator: No action needed, ending")
        return {"messages": [AIMessage(content="No processing needed")]}

    def _should_continue(self, state: MessagesState) -> str:
        """Decide whether to continue with more tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        print(f"🤔 SHOULD_CONTINUE: Last message type: {type(last_message).__name__}")
        if hasattr(last_message, 'content'):
            print(f"🤔 SHOULD_CONTINUE: Content: {(last_message.content[:50] + '...') if last_message.content and len(last_message.content) > 50 else last_message.content}")

        logger.info(f"Should continue: Last message type: {type(last_message).__name__}")
        if hasattr(last_message, 'content'):
            logger.info(f"Should continue: Content preview: {(last_message.content[:50] + '...') if last_message.content and len(last_message.content) > 50 else last_message.content}")

        # Continue if we have tool calls to execute
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("Should continue: YES - Tool calls pending execution")
            return "continue"

        # Check for completion messages - these should end
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content
            if content in ["Voice processing complete", "No processing needed", "Error in processing"]:
                logger.info(f"Should continue: NO - Process ended with: {content}")
                return "end"

        # Check if we have an audio file in any message - this means completion
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and msg.content.startswith("AUDIO_FILE:"):
                logger.info("Should continue: NO - AUDIO_FILE found, process complete")
                return "end"

        # If last message is a tool result that needs further processing, continue
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content
            if content.startswith("TRANSCRIPT:") or content.startswith("LLM_RESPONSE:"):
                logger.info(f"Should continue: YES - Tool result needs processing: {content.split(':')[0]}")
                return "continue"
            elif content.startswith("ERROR:"):
                logger.info("Should continue: NO - Error occurred")
                return "end"

        # Default to end to prevent infinite loops
        logger.info("Should continue: NO - Default to end (safety)")
        return "end"

    def process_voice_conversation(self, audio_data: bytes, filename: str = "audio.wav", thread_id: str = "default") -> str:
        """Process a voice conversation through the intelligent agent."""
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"Processing voice conversation: {filename}")

        # Encode audio data as base64 to include in message content
        import base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        # Create initial message with encoded audio data in content
        initial_message = HumanMessage(
            content=f"AUDIO_DATA:{audio_b64}|{filename}"
        )

        # Add recursion limit to config - increase for debugging
        config.update({"recursion_limit": 10})

        result = self.graph.invoke(
            {"messages": [initial_message]},
            config=config
        )

        # Extract the final audio file path from the results
        for message in reversed(result["messages"]):
            if hasattr(message, 'content') and message.content and message.content.startswith("AUDIO_FILE:"):
                audio_path = message.content.replace("AUDIO_FILE: ", "")
                logger.info(f"Returning audio path: {audio_path}")
                return audio_path

        # Fallback
        logger.warning("No audio file generated")
        return ""


# Global instance
voice_agent = None


def get_voice_agent() -> IntelligentVoiceAgent:
    """Get or create the global voice agent instance."""
    global voice_agent
    if voice_agent is None:
        voice_agent = IntelligentVoiceAgent()
    return voice_agent