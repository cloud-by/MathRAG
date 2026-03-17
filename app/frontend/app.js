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

function stageToZh(stage) {
  const mapping = {
    primary: '小学',
    junior_secondary: '初中',
    senior_secondary: '高中',
    undergraduate: '大学'
  };
  return mapping[String(stage || '').trim()] || String(stage || '未标注');
}

function difficultyToZh(difficulty) {
  const mapping = {
    easy: '简单',
    medium: '中等',
    hard: '困难'
  };
  return mapping[String(difficulty || '').trim()] || String(difficulty || '未标注');
}

function normalizeStringArray(value) {
  if (!Array.isArray(value)) {
    if (typeof value === 'string' && value.trim()) {
      return [value.trim()];
    }
    return [];
  }

  const result = [];
  const seen = new Set();

  value.forEach((item) => {
    const text = String(item ?? '').trim();
    if (!text || seen.has(text)) {
      return;
    }
    seen.add(text);
    result.push(text);
  });

  return result;
}

function scrollChatToBottom() {
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

function appendMessage(role, content, extraClass = '') {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'} ${extraClass}`.trim();

  const roleLabel = role === 'user' ? '你' : '助手';
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

  if (loading) {
    statusTextEl.textContent = '正在生成回答…';
  }
}

function renderAnswer(data) {
  const answer = data?.answer || '未返回答案。';
  const steps = normalizeStringArray(data?.steps);
  const usedKnowledge = normalizeStringArray(data?.used_knowledge);

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
  answerBoxEl.className = 'result-card';
  answerBoxEl.innerHTML = html;
}

function buildReferenceMeta(item) {
  const parts = [];

  if (item.stage) {
    parts.push(`学段：${stageToZh(item.stage)}`);
  }
  if (item.course) {
    parts.push(`课程：${escapeHtml(item.course)}`);
  }
  if (item.category) {
    parts.push(`类别：${escapeHtml(item.category)}`);
  }
  if (item.difficulty) {
    parts.push(`难度：${difficultyToZh(item.difficulty)}`);
  }

  return parts.length > 0
    ? `<div class="ref-meta">${parts.join(' ｜ ')}</div>`
    : '';
}

function buildReferenceExtraMeta(item) {
  const parts = [];

  if (item.chunk_id) {
    parts.push(`chunk_id：${escapeHtml(item.chunk_id)}`);
  }
  if (item.source_id) {
    parts.push(`source_id：${escapeHtml(item.source_id)}`);
  }
  if (item.source_line) {
    parts.push(`source_line：${escapeHtml(item.source_line)}`);
  }

  return parts.length > 0
    ? `<div class="ref-meta">${parts.join(' ｜ ')}</div>`
    : '';
}

function renderReferences(references) {
  if (!Array.isArray(references) || references.length === 0) {
    referencesBoxEl.className = 'stack-list empty-state';
    referencesBoxEl.textContent = '这次回答没有返回参考知识。';
    return;
  }

  referencesBoxEl.className = 'stack-list';
  referencesBoxEl.innerHTML = references.map((item) => {
    const keywords = normalizeStringArray(item.keywords);
    const steps = normalizeStringArray(item.steps);
    const prerequisites = normalizeStringArray(item.prerequisites);

    const keywordsHtml = keywords.length > 0
      ? `
        <div class="answer-block-title">关键词</div>
        <div class="answer-tags">
          ${keywords.map(keyword => `<span class="tag">${escapeHtml(keyword)}</span>`).join('')}
        </div>
      `
      : '';

    const prerequisitesHtml = prerequisites.length > 0
      ? `
        <div class="answer-block-title">前置知识</div>
        <div class="answer-tags">
          ${prerequisites.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('')}
        </div>
      `
      : '';

    const contentText = item.answer_context || item.content || '';
    const contentHtml = contentText
      ? `<div class="ref-text">${nl2br(contentText)}</div>`
      : '<div class="ref-text">暂无知识内容。</div>';

    const exampleHtml = item.example
      ? `<div class="ref-example"><strong>示例：</strong>${nl2br(item.example)}</div>`
      : '';

    const stepsHtml = steps.length > 0
      ? `
        <div class="answer-block-title">参考步骤</div>
        <ol class="answer-list">
          ${steps.map(step => `<li>${escapeHtml(step)}</li>`).join('')}
        </ol>
      `
      : '';

    const score = Number(item.score ?? 0);
    const scoreText = Number.isFinite(score) ? score.toFixed(6) : '0.000000';

    return `
      <div class="ref-item">
        <div class="ref-header">
          <div class="ref-title">[${escapeHtml(item.rank)}] ${escapeHtml(item.title || '未命名知识点')}</div>
          <div class="ref-score">score=${scoreText}</div>
        </div>
        ${buildReferenceMeta(item)}
        ${buildReferenceExtraMeta(item)}
        ${keywordsHtml}
        ${prerequisitesHtml}
        <div class="answer-block-title">知识内容</div>
        ${contentHtml}
        ${exampleHtml}
        ${stepsHtml}
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
    const text = String(question ?? '').trim();
    if (!text) {
      return;
    }

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'related-btn';
    btn.textContent = text;
    btn.addEventListener('click', () => {
      if (isLoading) {
        return;
      }
      questionInput.value = text;
      questionInput.focus();
      sendQuestion(text);
    });
    relatedBoxEl.appendChild(btn);
  });

  if (!relatedBoxEl.children.length) {
    relatedBoxEl.className = 'related-list empty-state';
    relatedBoxEl.textContent = '这次回答没有返回相关推荐问题。';
  }
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

  const loadingMessageEl = appendMessage('assistant', '正在思考…', 'loading-dots');
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

    const answerText = data?.answer || '未返回答案。';
    appendMessage('assistant', answerText);
    history.push({ role: 'assistant', content: answerText });

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
      <div class="message-role">助手</div>
      <div class="message-content">
        你好，欢迎来到 MathRAG。你可以先试试这些问题：<br>
        1. <code>x^2+4x+3=0 怎么解？</code><br>
        2. <code>平方差公式是什么？</code><br>
        3. <code>为什么这题可以因式分解？</code><br>
        4. <code>导数的几何意义是什么？</code>
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