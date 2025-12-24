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

// Chat behavior
const messageForm = document.getElementById('messageForm');
const questionInput = document.getElementById('question');
const chatContainer = document.getElementById('chatContainer');
const typingIndicator = document.getElementById('typingIndicator');
const sendBtn = document.getElementById('sendBtn');
let isWaiting = false;

function appendMessage(text, who = 'ai'){
  const el = document.createElement('div');
  el.className = 'message ' + (who === 'user' ? 'user':'ai');
  el.innerHTML = `<div class="content">${escapeHtml(text)}</div>`;
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
    const res = await fetch('/ask',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:text})
    });
    const data = await res.json();
    appendMessage(data.answer || 'No answer received', 'ai');
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

