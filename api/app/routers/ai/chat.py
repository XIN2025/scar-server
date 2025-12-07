from fastapi import APIRouter, HTTPException, Form, WebSocket, WebSocketDisconnect, Request, Query, Depends
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState
import json
import logging
import asyncio
import base64
from typing import Optional, Dict

from app.services.ai_services.chat_service import get_chat_service
from app.services.ai_services.mongodb_vectorstore import get_vector_store
from app.services.ai_services.transcription_service import get_transcription_service
from app.core.security import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])

@chat_router.post("/send")
async def send_message(
    message: str = Form(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Send a chat message and get a context-aware response from the RAG system.
    """
    try:
        chat_service = get_chat_service()
        result = await chat_service.chat(message, current_user.email)
        
        if not result.get("success", False):
             raise HTTPException(status_code=500, detail=result.get("error", "Chat processing failed."))

        return {
            "success": result["success"],
            "response": result["response"],
            "follow_up_questions": result.get("follow_up_questions", []),
            "context_used": result.get("context_used", 0)
        }
    except Exception as e:
        logger.error(f"Error in /chat/send endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the chat endpoint: {str(e)}")

@chat_router.get("/stream")
async def stream_chat(
    message: str,
    conversation_history: Optional[str] = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Stream a chat response in real-time using Server-Sent Events (SSE).
    """
    try:
        history = []
        if conversation_history:
            try:
                history = json.loads(conversation_history)
            except Exception as e:
                logger.warning(f"Failed to parse conversation history: {e}")
                history = []
        
        chat_service = get_chat_service()
        async def event_generator():
            try:
                async for event in chat_service.chat_stream(message, current_user.email, conversation_history=history):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Error in event generator: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred during streaming'})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache", 
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "X-Accel-Buffering": "no"   
            }
        )
    except Exception as e:
        logger.error(f"Error in /chat/stream endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during streaming: {str(e)}")

@chat_router.options("/stream")
async def stream_chat_options():
    """
    Handle CORS preflight requests for the stream endpoint.
    """
    return {"message": "OK"}

@chat_router.get("/vector-store-stats")
async def get_vector_store_stats():
    """
    Get statistics about the MongoDB vector store.
    """
    try:
        vector_store = get_vector_store()
        vector_stats = await asyncio.to_thread(vector_store.get_stats)
        
        return {
            "vector_store": vector_stats
        }
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@chat_router.delete("/clear-vector-store")
async def clear_vector_store():
    """
    DANGER: Clears all data from the MongoDB vector store.
    This will delete all learned knowledge from all uploaded documents for all users.
    """
    try:
        vector_store = get_vector_store()
        result = await asyncio.to_thread(vector_store.collection.delete_many, {})
        
        logger.warning(f"Cleared {result.deleted_count} documents from the vector store.")
        
        return {
            "success": True,
            "message": f"Vector store cleared successfully. {result.deleted_count} documents deleted."
        }
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")

@chat_router.websocket("/transcribe-stream")
async def transcribe_stream(websocket: WebSocket):
 
    try:
        await websocket.accept()
        print("🎤 [TRANSCRIBE] WebSocket connection accepted for transcription")
        logger.info("🎤 WebSocket connection accepted for transcription")
    except Exception as e:
        print(f"❌ [TRANSCRIBE] Failed to accept WebSocket: {e}")
        logger.error(f"❌ Failed to accept WebSocket: {e}", exc_info=True)
        return
    
    transcription_service = None
    
    try:
        print("🔧 [TRANSCRIBE] Creating Whisper transcription service instance...")
        logger.info("🔧 Creating Whisper transcription service instance...")
        transcription_service = get_transcription_service()
        print("✅ [TRANSCRIBE] Whisper transcription service instance created")
        logger.info("✅ Whisper transcription service instance created")
        
        cumulative_interim = ""
        finalized_segments = []
        
        async def safe_send_json(message: dict):
            """Safely send JSON message through websocket, checking connection state first"""
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.debug("WebSocket not connected, skipping message send")
                    return False
                await websocket.send_json(message)
                return True
            except WebSocketDisconnect:
                logger.debug("WebSocket disconnected, cannot send message")
                return False
            except Exception as e:
                logger.warning(f"Error sending message through websocket: {e}")
                return False
        
        async def on_message(is_final: bool, text: str, extra_data: Dict = None):
            nonlocal cumulative_interim, finalized_segments
            try:
                if extra_data and extra_data.get("action") == "auto_stop":
                    logger.info("🔇 Auto-stop signal received due to silence")
                    await safe_send_json({
                        "action": "auto_stop",
                        "is_final": True,
                        "text": "",
                        "reason": "silence_detected"
                    })
                    return
                
                if text and text.strip():
                    print(f"📥 [CHAT] Received from transcription service: '{text}' (is_final: {is_final})")
                    logger.info(f"📥 Received from transcription service: '{text}' (is_final: {is_final})")
                    
                    if is_final:
                        # For final results, add to finalized segments
                        finalized_segments.append(text.strip())
                        response_message = {
                            "is_final": True,
                            "text": text,
                            "action": "final"
                        }
                        print(f"📤 [CHAT] Sending final to frontend: {response_message}")
                        await safe_send_json(response_message)
                        # Clear interim text since it's now finalized
                        cumulative_interim = ""
                        logger.info(f"📝 Transcription (final): '{text}' - Total finalized: {len(finalized_segments)}")
                    else:
                        # For interim results, update the interim text
                        cumulative_interim = text
                        response_message = {
                            "is_final": False,
                            "text": text,
                            "action": "interim"
                        }
                        print(f"📤 [CHAT] Sending interim to frontend: {response_message}")
                        await safe_send_json(response_message)
                        logger.info(f"📝 Transcription (interim): '{text}'")
                    
            except Exception as e:
                logger.error(f"❌ Error processing Whisper message: {e}", exc_info=True)
        
        async def on_error(error: str):
            logger.error(f"❌ Whisper error: {error}")
            await safe_send_json({
                "error": str(error),
                "is_final": False,
                "text": ""
            })
        
        print("🔌 [TRANSCRIBE] Attempting to create real-time Whisper connection...")
        logger.info("🔌 Attempting to create real-time Whisper connection...")
        connection_success = await transcription_service.create_connection(
            on_message_callback=on_message,
            on_error_callback=on_error
        )
        
        if not connection_success:
            print("❌ [TRANSCRIBE] Failed to establish Whisper connection")
            logger.error("❌ Failed to establish Whisper connection")
            await safe_send_json({
                "error": "Failed to connect to transcription service",
                "is_final": False,
                "text": ""
            })
            try:
                await websocket.close()
            except:
                pass
            return
        
        print("✅ [TRANSCRIBE] Real-time transcription service ready, waiting for audio chunks from frontend...")
        logger.info("✅ Real-time transcription service ready, waiting for audio chunks from frontend...")
        
        audio_chunks_received = 0
        while True:
            try:
                message = await websocket.receive_json()
                print(f"📥 [TRANSCRIBE] Received message from frontend: type={message.get('type')}")
                logger.info(f"📥 Received message from frontend: type={message.get('type')}")
                
                if message.get("type") == "stop":
                    print(f"🛑 [TRANSCRIBE] Stop signal received after {audio_chunks_received} audio chunks")
                    logger.info(f"🛑 Stop signal received after {audio_chunks_received} audio chunks")
                    
                    print("⏳ [TRANSCRIBE] Processing final audio chunks...")
                    try:
                        await transcription_service.finalize_stream()
                        # Small delay to ensure final processing completes
                        await asyncio.sleep(0.1)
                        print("✅ [TRANSCRIBE] Final processing complete")
                    except Exception as e:
                        print(f"⚠️ [TRANSCRIBE] Error in final processing: {e}")
                        logger.warning(f"⚠️ Error in final processing: {e}")
                    
                    print("🔌 [TRANSCRIBE] Closing connection gracefully")
                    break
                
                if message.get("type") == "audio":
                    audio_base64 = message.get("data") or message.get("audio") or ""
                    print(f"🔍 [TRANSCRIBE] Audio data length: {len(audio_base64) if audio_base64 else 0}")
                    
                    if audio_base64:
                        audio_chunks_received += 1
                        try:
                            audio_bytes = base64.b64decode(audio_base64)
                            print(f"🎵 [TRANSCRIBE] Real-time audio chunk {audio_chunks_received}: {len(audio_bytes)} bytes")
                            
                            await transcription_service.send_audio(audio_bytes)
                            print(f"✅ [TRANSCRIBE] Audio chunk {audio_chunks_received} queued for real-time processing")
                        except Exception as e:
                            print(f"❌ [TRANSCRIBE] Error decoding audio chunk {audio_chunks_received}: {e}")
                            logger.error(f"❌ Error decoding audio chunk {audio_chunks_received}: {e}")
                    else:
                        print("⚠️ [TRANSCRIBE] Received empty audio data")
                        logger.warning("⚠️ Received empty audio data")
                        
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected by client after {audio_chunks_received} chunks")
                break
            except Exception as e:
                logger.error(f"❌ Error receiving/processing audio: {e}", exc_info=True)
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        # Try to send error message if connection is still open
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "error": str(e),
                    "is_final": False,
                    "text": ""
                })
        except:
            pass
    
    finally:
        if transcription_service:
            await transcription_service.close_connection()
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info("✅ Transcription WebSocket closed")
