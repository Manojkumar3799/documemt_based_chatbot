// Preserve upload behavior
const uploadForm = document.getElementById('uploadForm');
if (uploadForm) {
  uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const status = document.getElementById('uploadStatus');
    const formData = new FormData(this);
    status.textContent = 'Uploading...';

    fetch('/upload', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => {
        status.textContent = data.message || 'Uploaded';
      })
      .catch(() => { status.textContent = 'Upload failed'; });
  });
}

// Chat behavior with session support
const messageForm = document.getElementById('messageForm');
const questionInput = document.getElementById('question');
const chatContainer = document.getElementById('chatContainer');
const typingIndicator = document.getElementById('typingIndicator');
const sendBtn = document.getElementById('sendBtn');
let isWaiting = false;

// session handling: store a session_id locally so the server keeps chat history per session
let sessionId = localStorage.getItem('docbot_session') || null;

function appendMessage(text, who = 'ai', meta = null){
  const el = document.createElement('div');
  el.className = 'message ' + (who === 'user' ? 'user':'ai');
  el.innerHTML = `<div class="content">${escapeHtml(text)}</div>`;
  if(meta && meta.length){
    const footer = document.createElement('div');
    footer.className = 'message-meta';
    footer.innerHTML = '<small class="muted">Sources: ' + meta.map(m => escapeHtml(m.source)).join(', ') + '</small>';
    el.appendChild(footer);
  }
  chatContainer.appendChild(el);
  scrollToBottom();
}

function scrollToBottom(){
  requestAnimationFrame(()=>{ chatContainer.scrollTop = chatContainer.scrollHeight; });
}

function showTyping(show){
  if(!typingIndicator) return;
  typingIndicator.hidden = !show;
}

// safely escape user text
function escapeHtml(str){
  if(!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/\'/g,'&#039;').replace(/\n/g,'<br>');
}

// Send request and handle messages
async function sendMessage(){
  if(isWaiting) return;
  const text = questionInput.value.trim();
  if(!text) return;

  appendMessage(text, 'user');
  questionInput.value = '';
  adjustTextareaHeight(questionInput);

  // show typing
  isWaiting = true;
  showTyping(true);
  sendBtn.disabled = true;

  try{
    const res = await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:text, session_id: sessionId})
    });
    const data = await res.json();

    // If server returned an error, display it to the user
    if(!res.ok){
      appendMessage(data.error || 'Server error', 'ai');
      return;
    }

    if(data.error){
      appendMessage(data.error, 'ai');
      return;
    }

    // Save the session returned by server (so a new session gets persistently stored)
    if(data.session_id){
      sessionId = data.session_id;
      localStorage.setItem('docbot_session', sessionId);
    }

    appendMessage(data.answer || 'No answer received', 'ai', data.sources || []);
  }catch(err){
    appendMessage('Error retrieving answer', 'ai');
  }finally{
    isWaiting = false;
    showTyping(false);
    sendBtn.disabled = false;
  }
}

if(messageForm){
  messageForm.addEventListener('submit',(e)=>{ e.preventDefault(); sendMessage(); });
}

// enter to send, shift+enter for newline
if(questionInput){
  questionInput.addEventListener('keydown',(e)=>{
    if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }
  });
  // auto-expand
  questionInput.addEventListener('input', ()=> adjustTextareaHeight(questionInput));
  // ensure initial size
  adjustTextareaHeight(questionInput);
}

function adjustTextareaHeight(el){
  if(!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

// init: small welcome message
if(chatContainer && chatContainer.children.length === 0){
  appendMessage('Hi 👋 — upload a PDF to get started. Ask questions about the document and I will answer.', 'ai');
}

// show session info in the UI
const sessionInfoEl = document.getElementById('sessionId');
if(sessionId && sessionInfoEl){ sessionInfoEl.textContent = sessionId; }

// clear session button behavior
const clearBtn = document.getElementById('clearSessionBtn');
if(clearBtn){
  clearBtn.addEventListener('click', async ()=>{
    if(!sessionId){
      // nothing to clear; just reset UI
      localStorage.removeItem('docbot_session');
      if(sessionInfoEl) sessionInfoEl.textContent = '(new)';
      chatContainer.innerHTML = '';
      appendMessage('Session cleared — upload a PDF to start a new chat.','ai');
      return;
    }

    try{
      const res = await fetch('/clear_session',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({session_id: sessionId})
      });
      const data = await res.json();
      localStorage.removeItem('docbot_session');
      sessionId = null;
      if(sessionInfoEl) sessionInfoEl.textContent = '(new)';
      chatContainer.innerHTML = '';
      appendMessage(data.message || 'Session cleared.','ai');
    }catch(err){
      appendMessage('Failed to clear session','ai');
    }
  });
}
