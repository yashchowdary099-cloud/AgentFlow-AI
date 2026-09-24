/**
 * AgentFlow AI — Frontend Application Logic
 * Interacts with FastAPI backend, manages conversation sessions, PDF upload,
 * markdown rendering, and visual Agent Activity tracking.
 */

// Base API URL: Points to backend at port 8000 when frontend is served on port 5500 or other ports
const API_BASE = (window.location.port === '8000') ? '' : 'https://agentflow-ai-sjsi.onrender.com';

// Application State
const state = {
  conversationId: null,
  isGenerating: false,
  activeDocument: null,
  conversations: []
};

// DOM Element References
const elements = {
  chatContainer: document.getElementById('chat-messages-container'),
  messagesList: document.getElementById('messages-list'),
  welcomeScreen: document.getElementById('welcome-screen'),
  chatForm: document.getElementById('chat-form'),
  userInput: document.getElementById('user-input'),
  sendBtn: document.getElementById('send-btn'),
  attachBtn: document.getElementById('attach-btn'),
  pdfFileInput: document.getElementById('pdf-file-input'),
  newChatBtn: document.getElementById('new-chat-btn'),
  mobileNewChatBtn: document.getElementById('mobile-new-chat-btn'),
  mobileMenuBtn: document.getElementById('mobile-menu-btn'),
  sidebar: document.getElementById('sidebar'),
  sidebarBackdrop: document.getElementById('sidebar-backdrop'),
  conversationsList: document.getElementById('conversations-list'),

  // Document status elements
  documentCard: document.getElementById('document-card'),
  docName: document.getElementById('doc-name'),
  docMeta: document.getElementById('doc-meta'),
  removeDocBtn: document.getElementById('remove-doc-btn'),
  uploadProgressContainer: document.getElementById('upload-progress-container'),
  uploadProgressFill: document.getElementById('upload-progress-fill'),
  uploadStatusText: document.getElementById('upload-status-text'),

  // Header / indicator elements
  apiStatusDot: document.getElementById('api-status-dot'),
  apiStatusLabel: document.getElementById('api-status-label'),
  headerStatus: document.getElementById('header-status')
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  initAutoResizeTextarea();
  await checkHealth();
  await checkDocumentStatus();
  await loadConversations();
  await startNewChat(false); // Initialize session without wiping loaded list
});

function setupEventListeners() {
  // Chat form submit
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  // Textarea keyboard shortcuts: Enter = send, Shift+Enter = newline
  elements.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // New Chat buttons
  elements.newChatBtn.addEventListener('click', () => startNewChat(true));
  if (elements.mobileNewChatBtn) {
    elements.mobileNewChatBtn.addEventListener('click', () => {
      startNewChat(true);
      closeSidebar();
    });
  }

  // File Upload Handlers
  elements.attachBtn.addEventListener('click', () => elements.pdfFileInput.click());
  elements.pdfFileInput.addEventListener('change', handleFileUpload);
  elements.removeDocBtn.addEventListener('click', handleRemoveDocument);

  // Suggestion card clicks
  document.querySelectorAll('.suggestion-card').forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt) {
        elements.userInput.value = prompt;
        elements.userInput.focus();
        handleSendMessage();
      }
    });
  });

  // Mobile navigation
  if (elements.mobileMenuBtn) {
    elements.mobileMenuBtn.addEventListener('click', toggleSidebar);
  }
  if (elements.sidebarBackdrop) {
    elements.sidebarBackdrop.addEventListener('click', closeSidebar);
  }
}

// Auto-expand textarea as user types
function initAutoResizeTextarea() {
  elements.userInput.addEventListener('input', () => {
    elements.userInput.style.height = 'auto';
    elements.userInput.style.height = Math.min(elements.userInput.scrollHeight, 160) + 'px';
  });
}

function toggleSidebar() {
  elements.sidebar.classList.toggle('open');
  elements.sidebarBackdrop.classList.toggle('active');
}

function closeSidebar() {
  elements.sidebar.classList.remove('open');
  elements.sidebarBackdrop.classList.remove('active');
}

// ============================================================================
// API Calls & Health
// ============================================================================
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === 'ok') {
      elements.apiStatusDot.className = 'status-dot online';
      elements.apiStatusLabel.textContent = data.gemini_configured ? 'Agent Ready (Gemini)' : 'Agent Ready (Local)';
      elements.headerStatus.textContent = data.gemini_configured ? 'Autonomous Agent Online' : 'Agent Ready (Add API Key in .env)';
    }
  } catch (err) {
    elements.apiStatusDot.className = 'status-dot offline';
    elements.apiStatusLabel.textContent = 'Server Offline';
    elements.headerStatus.textContent = 'Backend Not Reachable';
  }
}

async function checkDocumentStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/document`);
    const data = await res.json();
    updateDocumentUI(data);
  } catch (err) {
    console.warn('Could not fetch document status:', err);
  }
}

function updateDocumentUI(doc) {
  state.activeDocument = doc;
  if (doc && doc.has_document) {
    elements.documentCard.className = 'document-card has-doc';
    elements.docName.textContent = doc.filename;
    elements.docMeta.textContent = `${doc.num_pages} pages • ${doc.num_chunks} chunks • Ready`;
    elements.removeDocBtn.style.display = 'block';
  } else {
    elements.documentCard.className = 'document-card empty-doc';
    elements.docName.textContent = 'No PDF uploaded';
    elements.docMeta.textContent = 'Upload a PDF to ask questions';
    elements.removeDocBtn.style.display = 'none';
  }
}

// ============================================================================
// Chat Handling
// ============================================================================
async function handleSendMessage() {
  const message = elements.userInput.value.trim();
  if (!message || state.isGenerating) return;

  // Clear input
  elements.userInput.value = '';
  elements.userInput.style.height = 'auto';

  // Hide welcome screen if visible
  if (elements.welcomeScreen.style.display !== 'none') {
    elements.welcomeScreen.style.display = 'none';
  }

  // Append user message to UI
  appendUserMessage(message);

  // Set generating state
  state.isGenerating = true;
  elements.sendBtn.disabled = true;

  // Determine dynamic thinking label based on query heuristics
  const lower = message.toLowerCase();
  let thinkingLabel = 'AI Agent is working...';
  if (/[\d\+\-\*\/\%\^]/.test(lower) && (lower.includes('calculate') || lower.includes('%') || lower.includes('*'))) {
    thinkingLabel = 'Analyzing math expression with Calculator...';
  } else if (lower.includes('pdf') || lower.includes('document') || lower.includes('summarize')) {
    thinkingLabel = 'Searching document & retrieving context...';
  }

  // Show thinking placeholder
  const thinkingId = appendThinkingBubble(thinkingLabel);
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        conversation_id: state.conversationId
      })
    });

    if (!res.ok) {
      throw new Error(`HTTP error ${res.status}`);
    }

    const data = await res.json();

    // Update active conversation ID
    if (data.conversation_id) {
      state.conversationId = data.conversation_id;
    }

    // Remove thinking indicator
    removeElement(thinkingId);

    // Append AI response
    appendAssistantMessage(data.answer, data.tool_used, data.activity);

    // Reload sidebar conversations to reflect new title/activity
    loadConversations();
  } catch (err) {
    console.error('Chat error:', err);
    removeElement(thinkingId);
    appendAssistantMessage(
      "Something went wrong while generating the answer. Please check that the backend server is running.",
      "none",
      ["Request received", "Connection error encountered"]
    );
  } finally {
    state.isGenerating = false;
    elements.sendBtn.disabled = false;
    elements.userInput.focus();
    scrollToBottom();
  }
}

// ============================================================================
// UI Rendering Functions
// ============================================================================
function appendUserMessage(text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message-wrapper user';

  msgDiv.innerHTML = `
    <div class="avatar" title="You">
      <i class="fa-solid fa-user"></i>
    </div>
    <div class="message-bubble">
      <div class="message-content">${escapeHTML(text)}</div>
    </div>
  `;

  elements.messagesList.appendChild(msgDiv);
}

function appendAssistantMessage(text, toolUsed = 'none', activitySteps = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message-wrapper assistant';

  // Tool badge styling
  let badgeHTML = '';
  if (toolUsed === 'calculator') {
    badgeHTML = `<div class="tool-badge calculator"><i class="fa-solid fa-calculator"></i> Tool: Smart Calculator</div>`;
  } else if (toolUsed === 'document') {
    badgeHTML = `<div class="tool-badge document"><i class="fa-solid fa-book-open"></i> Tool: PDF Document RAG</div>`;
  } else if (toolUsed === 'web_search') {
    badgeHTML = `<div class="tool-badge web_search"><i class="fa-solid fa-globe"></i> Tool: Web Search</div>`;
  } else {
    badgeHTML = `<div class="tool-badge none"><i class="fa-solid fa-bolt"></i> Direct Gemini Response</div>`;
  }

  // Agent Activity Visual Panel
  let activityHTML = '';
  if (activitySteps && activitySteps.length > 0) {
    const stepsHTML = activitySteps.map(step => `
      <li class="activity-step">
        <i class="fa-solid fa-check"></i>
        <span>${escapeHTML(step)}</span>
      </li>
    `).join('');

    activityHTML = `
      <div class="agent-activity-box">
        <div class="activity-title">
          <i class="fa-solid fa-timeline"></i> Agent Activity
        </div>
        <ul class="activity-steps">
          ${stepsHTML}
        </ul>
      </div>
    `;
  }

  // Markdown rendering
  const parsedContent = (typeof marked !== 'undefined') ? marked.parse(text) : escapeHTML(text);

  msgDiv.innerHTML = `
    <div class="avatar" title="AgentFlow AI">
      <i class="fa-solid fa-robot"></i>
    </div>
    <div class="message-bubble">
      ${badgeHTML}
      ${activityHTML}
      <div class="message-content">${parsedContent}</div>
      <div class="message-actions">
        <button class="copy-btn" title="Copy response">
          <i class="fa-regular fa-copy"></i>
          <span>Copy</span>
        </button>
      </div>
    </div>
  `;

  // Attach working copy button
  const copyBtn = msgDiv.querySelector('.copy-btn');
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(text).then(() => {
      copyBtn.classList.add('copied');
      copyBtn.innerHTML = `<i class="fa-solid fa-check"></i> <span>Copied!</span>`;
      setTimeout(() => {
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = `<i class="fa-regular fa-copy"></i> <span>Copy</span>`;
      }, 2000);
    });
  });

  elements.messagesList.appendChild(msgDiv);
  scrollToBottom();
}

function appendThinkingBubble(label = 'AI Agent is working...') {
  const id = 'thinking-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'message-wrapper assistant';
  div.innerHTML = `
    <div class="avatar">
      <i class="fa-solid fa-robot"></i>
    </div>
    <div class="message-bubble">
      <div class="thinking-box">
        <div class="thinking-dots">
          <div class="dot"></div>
          <div class="dot"></div>
          <div class="dot"></div>
        </div>
        <span>${escapeHTML(label)}</span>
      </div>
    </div>
  `;
  elements.messagesList.appendChild(div);
  return id;
}

function removeElement(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ============================================================================
// PDF File Upload
// ============================================================================
async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a valid PDF file.');
    elements.pdfFileInput.value = '';
    return;
  }

  // Show progress animation
  elements.uploadProgressContainer.style.display = 'block';
  elements.uploadProgressFill.style.width = '30%';
  elements.uploadStatusText.textContent = 'Uploading PDF...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    elements.uploadProgressFill.style.width = '65%';
    elements.uploadStatusText.textContent = 'Extracting text & chunking...';

    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });

    elements.uploadProgressFill.style.width = '90%';
    elements.uploadStatusText.textContent = 'Indexing vector embeddings...';

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Upload failed');
    }

    elements.uploadProgressFill.style.width = '100%';
    elements.uploadStatusText.textContent = 'Ready!';

    setTimeout(() => {
      elements.uploadProgressContainer.style.display = 'none';
      elements.uploadProgressFill.style.width = '0%';
    }, 1200);

    // Update Sidebar Document Card
    updateDocumentUI({
      has_document: true,
      filename: data.filename,
      num_pages: data.pages,
      num_chunks: data.chunks
    });

    // Notify in chat
    if (elements.welcomeScreen.style.display !== 'none') {
      elements.welcomeScreen.style.display = 'none';
    }

    appendAssistantMessage(
      `📄 **Document Uploaded Successfully!**\n\n` +
      `**Filename**: \`${data.filename}\`\n` +
      `**Pages Processed**: ${data.pages}\n` +
      `**Chunks Created**: ${data.chunks}\n\n` +
      `You can now ask questions about this document, such as:\n` +
      `- *"Summarize this document"*\n` +
      `- *"What are the main takeaways?"*\n` +
      `- *"Explain the key points in simple English"*`,
      'document',
      ['Upload validated', 'Text extracted page-by-page', 'Text chunked & vectorized', 'Document index ready']
    );

  } catch (err) {
    console.error('Upload error:', err);
    elements.uploadProgressContainer.style.display = 'none';
    alert('PDF upload failed: ' + err.message);
  } finally {
    elements.pdfFileInput.value = '';
  }
}

async function handleRemoveDocument() {
  if (!confirm('Remove the currently active PDF document?')) return;
  try {
    await fetch(`${API_BASE}/api/document`, { method: 'DELETE' });
    updateDocumentUI({ has_document: false });
    appendAssistantMessage(
      'The active PDF document has been removed. You can upload a new document whenever you like.',
      'none',
      ['Active document cleared']
    );
  } catch (err) {
    console.error('Error removing document:', err);
  }
}

// ============================================================================
// Session & Conversation Management
// ============================================================================
async function startNewChat(shouldCreateOnBackend = true) {
  // Clear UI messages
  elements.messagesList.innerHTML = '';
  elements.welcomeScreen.style.display = 'flex';

  if (shouldCreateOnBackend) {
    try {
      const res = await fetch(`${API_BASE}/api/new-chat`, { method: 'POST' });
      const data = await res.json();
      state.conversationId = data.conversation_id;
      await loadConversations();
    } catch (err) {
      console.error('New chat creation error:', err);
      state.conversationId = 'session-' + Date.now();
    }
  } else {
    state.conversationId = 'session-' + Date.now();
  }

  // Highlight active session
  highlightActiveConversation();
}

async function loadConversations() {
  try {
    const res = await fetch(`${API_BASE}/api/conversations`);
    const list = await res.json();
    state.conversations = list;
    renderConversationsList(list);
  } catch (err) {
    console.warn('Could not load conversations:', err);
  }
}

function renderConversationsList(conversations) {
  if (!conversations || conversations.length === 0) {
    elements.conversationsList.innerHTML = `<div class="empty-history-text">No previous chats</div>`;
    return;
  }

  elements.conversationsList.innerHTML = conversations.map(conv => `
    <div class="conversation-item ${conv.id === state.conversationId ? 'active' : ''}" data-id="${conv.id}">
      <span class="conv-title" title="${escapeHTML(conv.title || 'Untitled')}">
        <i class="fa-regular fa-message"></i> ${escapeHTML(conv.title || 'Untitled')}
      </span>
      <button class="conv-delete-btn" data-id="${conv.id}" title="Delete chat">
        <i class="fa-solid fa-trash-can"></i>
      </button>
    </div>
  `).join('');

  // Attach click events
  elements.conversationsList.querySelectorAll('.conversation-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('.conv-delete-btn')) return;
      const cid = item.getAttribute('data-id');
      loadConversationHistory(cid);
    });
  });

  elements.conversationsList.querySelectorAll('.conv-delete-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const cid = btn.getAttribute('data-id');
      if (confirm('Delete this conversation?')) {
        await deleteConversation(cid);
      }
    });
  });
}

function highlightActiveConversation() {
  elements.conversationsList.querySelectorAll('.conversation-item').forEach(item => {
    if (item.getAttribute('data-id') === state.conversationId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
}

async function loadConversationHistory(conversationId) {
  try {
    const res = await fetch(`${API_BASE}/api/history/${conversationId}`);
    const messages = await res.json();

    state.conversationId = conversationId;
    elements.messagesList.innerHTML = '';

    if (messages.length > 0) {
      elements.welcomeScreen.style.display = 'none';
      messages.forEach(msg => {
        if (msg.role === 'user') {
          appendUserMessage(msg.content);
        } else if (msg.role === 'assistant') {
          appendAssistantMessage(msg.content, msg.tool_used, msg.activity);
        }
      });
    } else {
      elements.welcomeScreen.style.display = 'flex';
    }

    highlightActiveConversation();
    closeSidebar();
    scrollToBottom();
  } catch (err) {
    console.error('Error loading history:', err);
  }
}

async function deleteConversation(conversationId) {
  try {
    await fetch(`${API_BASE}/api/conversations/${conversationId}`, { method: 'DELETE' });
    if (state.conversationId === conversationId) {
      await startNewChat(true);
    } else {
      await loadConversations();
    }
  } catch (err) {
    console.error('Error deleting conversation:', err);
  }
}
