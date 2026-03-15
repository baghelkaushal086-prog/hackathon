async function sendMessage() {
    const userText = document.getElementById('user-input').value;
    
    const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: "candidate_01",
            message: userText,
            role_info: "Junior C++ Developer"
        })
    });

    const data = await response.json();
    document.getElementById('chat-box').innerHTML += `<p>AI: ${data.reply}</p>`;
}