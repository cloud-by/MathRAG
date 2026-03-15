const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const topKSelect = document.getElementById('top-k-select');
const chatHistoryEl = document.getElementById('chat-history');
const answerBoxEl = document.getElementById('answer-box');
const referencesBoxEl = document.getElementById('references-box');
const relatedBoxEl = document.getElementById('related-box');
const statusTextEl = document.getElementById('status-text');

const history = [];
let isLoading = false;

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function nl2br(value) {
  return escapeHtml(value).replace(/\n/g, '<br>');
}

function scrollChatToBottom() {
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

function appendMessage(role, content, extraClass = '') {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'} ${extraClass}`.trim();

  const roleLabel = role === 'user' ? '你' : '系统';
  wrapper.innerHTML = `
    <div class="message-role">${escapeHtml(roleLabel)}</div>
    <div class="message-content">${nl2br(content)}</div>
  `;

  chatHistoryEl.appendChild(wrapper);
  scrollChatToBottom();
  return wrapper;
}

function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  clearChatBtn.disabled = loading;
  questionInput.disabled = loading;
  topKSelect.disabled = loading;
  statusTextEl.textContent = loading ? '正在生成回答…' : '已完成';
}

function renderAnswer(data) {
  const answer = data?.answer || '未返回答案。';
  const steps = Array.isArray(data?.steps) ? data.steps : [];
  const usedKnowledge = Array.isArray(data?.used_knowledge) ? data.used_knowledge : [];

  let html = `<div class="answer-main">${nl2br(answer)}</div>`;

  if (steps.length > 0) {
    html += '<div class="answer-block-title">解题步骤</div>';
    html += '<ol class="answer-list">';
    html += steps.map(item => `<li>${escapeHtml(item)}</li>`).join('');
    html += '</ol>';
  }

  if (usedKnowledge.length > 0) {
    html += '<div class="answer-block-title">使用到的知识点</div>';
    html += '<div class="answer-tags">';
    html += usedKnowledge.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('');
    html += '</div>';
  }

  answerBoxEl.classList.remove('empty-state');
  answerBoxEl.innerHTML = html;
}

function renderReferences(references) {
  if (!Array.isArray(references) || references.length === 0) {
    referencesBoxEl.className = 'stack-list empty-state';
    referencesBoxEl.textContent = '这次回答没有返回参考知识。';
    return;
  }

  referencesBoxEl.className = 'stack-list';
  referencesBoxEl.innerHTML = references.map((item) => {
    const keywords = Array.isArray(item.keywords) && item.keywords.length > 0
      ? `<div class="ref-meta">关键词：${escapeHtml(item.keywords.join('，'))}</div>`
      : '';

    const example = item.example
      ? `<div class="ref-example"><strong>示例：</strong>${nl2br(item.example)}</div>`
      : '';

    return `
      <div class="ref-item">
        <div class="ref-header">
          <div class="ref-title">[${escapeHtml(item.rank)}] ${escapeHtml(item.title)}</div>
          <div class="ref-score">score=${Number(item.score ?? 0).toFixed(6)}</div>
        </div>
        <div class="ref-meta">类别：${escapeHtml(item.category)} ｜ chunk_id：${escapeHtml(item.chunk_id)}</div>
        ${keywords}
        <div class="ref-text">${nl2br(item.content || '')}</div>
        ${example}
      </div>
    `;
  }).join('');
}

function renderRelatedQuestions(questions) {
  if (!Array.isArray(questions) || questions.length === 0) {
    relatedBoxEl.className = 'related-list empty-state';
    relatedBoxEl.textContent = '这次回答没有返回相关推荐问题。';
    return;
  }

  relatedBoxEl.className = 'related-list';
  relatedBoxEl.innerHTML = '';

  questions.forEach((question) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'related-btn';
    btn.textContent = question;
    btn.addEventListener('click', () => {
      questionInput.value = question;
      questionInput.focus();
      sendQuestion(question);
    });
    relatedBoxEl.appendChild(btn);
  });
}

function renderError(message) {
  answerBoxEl.className = 'result-card';
  answerBoxEl.innerHTML = `<div class="answer-main">${nl2br(message)}</div>`;

  referencesBoxEl.className = 'stack-list empty-state';
  referencesBoxEl.textContent = '由于本次请求失败，暂无参考知识。';

  relatedBoxEl.className = 'related-list empty-state';
  relatedBoxEl.textContent = '由于本次请求失败，暂无相关推荐问题。';
}

async function sendQuestion(rawQuestion) {
  const question = String(rawQuestion || '').trim();
  if (!question || isLoading) {
    return;
  }

  appendMessage('user', question);
  history.push({ role: 'user', content: question });

  const loadingMessageEl = appendMessage('assistant', '正在思考', 'loading-dots');
  setLoading(true);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question,
        history,
        top_k: Number(topKSelect.value || 3)
      })
    });

    const data = await response.json();

    loadingMessageEl.remove();

    if (!response.ok) {
      const errorMessage = data?.detail || `请求失败，状态码：${response.status}`;
      appendMessage('assistant', `请求失败：${errorMessage}`, 'error-message');
      renderError(`请求失败：${errorMessage}`);
      statusTextEl.textContent = '请求失败';
      history.push({ role: 'assistant', content: `请求失败：${errorMessage}` });
      return;
    }

    appendMessage('assistant', data.answer || '未返回答案。');
    history.push({ role: 'assistant', content: data.answer || '未返回答案。' });

    renderAnswer(data);
    renderReferences(data.references || []);
    renderRelatedQuestions(data.related_questions || []);
    statusTextEl.textContent = '已完成';
  } catch (error) {
    loadingMessageEl.remove();
    const message = error instanceof Error ? error.message : '未知错误';
    appendMessage('assistant', `请求异常：${message}`, 'error-message');
    renderError(`请求异常：${message}`);
    statusTextEl.textContent = '请求异常';
    history.push({ role: 'assistant', content: `请求异常：${message}` });
  } finally {
    setLoading(false);
    questionInput.value = '';
    questionInput.focus();
  }
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  await sendQuestion(questionInput.value);
});

questionInput.addEventListener('keydown', async (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    await sendQuestion(questionInput.value);
  }
});

clearChatBtn.addEventListener('click', () => {
  if (isLoading) {
    return;
  }

  history.length = 0;
  chatHistoryEl.innerHTML = `
    <div class="message assistant-message welcome-card">
      <div class="message-role">系统</div>
      <div class="message-content">
        你好，欢迎来到 MathRAG。你可以先试试这些问题：<br>
        1. <code>x^2+4x+3=0 怎么解？</code><br>
        2. <code>平方差公式是什么？</code><br>
        3. <code>为什么这题可以因式分解？</code>
      </div>
    </div>
  `;

  answerBoxEl.className = 'result-card empty-state';
  answerBoxEl.textContent = '提交问题后，这里会显示答案。';

  referencesBoxEl.className = 'stack-list empty-state';
  referencesBoxEl.textContent = '提交问题后，这里会显示参考知识。';

  relatedBoxEl.className = 'related-list empty-state';
  relatedBoxEl.textContent = '提交问题后，这里会显示相关推荐问题。';

  statusTextEl.textContent = '待提问';
  questionInput.value = '';
  questionInput.focus();
});
