import google.generativeai as genai
import os

# 1. Setup API Key (Get yours at aistudio.google.com)
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Define the System Data (The Prompt Engineering)
role = "Junior Backend Developer"
skills = ["C++", "Data Structures", "SQL"]

system_instruction = f"""
You are an expert Technical Recruiter for a top tech firm. 
Your task is to interview a candidate for the {role} position.
Target Skills: {', '.join(skills)}.

Rules:
1. Start with a brief greeting and ask the first introductory question.
2. Ask ONLY ONE question at a time.
3. Wait for the candidate's response before asking the next question.
4. After 5 questions, say 'INTERVIEW_OVER' and provide a brief encouraging closing.
"""

# 3. Start the Chat Session
chat = model.start_chat(history=[])

print("--- AI Interviewer Started ---\n")

# Initial greeting/first question
response = chat.send_message(system_instruction)
print(f"AI: {response.text}")

# 4. Simple Interview Loop (For testing in your terminal)
for _ in range(5):
    user_input = input("You: ")
    response = chat.send_message(user_input)
    print(f"\nAI: {response.text}")
    
    if "INTERVIEW_OVER" in response.text:
        break