from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# This keeps the conversation memory for the session
chat_sessions = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    role_info: str # e.g., "C++ Developer"

@app.post("/chat")
async def interview_step(request: ChatRequest):
    # 1. Get or create a session for this user
    if request.user_id not in chat_sessions:
        chat_sessions[request.user_id] = model.start_chat(history=[])
        # Send initial system instructions hidden from the user
        chat_sessions[request.user_id].send_message(f"System: Start an interview for {request.role_info}")

    # 2. Send user message to AI
    response = chat_sessions[request.user_id].send_message(request.message)
    
    # 3. Return only the AI text to the Frontend
    return {"reply": response.text}