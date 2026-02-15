/**
 * Gift Advisor - Standalone frontend
 * Adapted from havanora-shopify hn_custom_script.txt
 */
(function () {
  const app = document.querySelector('[data-ga-app]');
  if (!app) return;

  const messagesEl = app.querySelector('[data-ga-messages]');
  const formEl = app.querySelector('[data-ga-form]');
  const inputEl = app.querySelector('[data-ga-input]');
  const sendBtn = app.querySelector('[data-ga-send]');
  const heroEl = app.querySelector('[data-ga-hero]');
  const occasionEl = app.querySelector('[data-ga-occasion]');
  const chipsEl = app.querySelector('[data-ga-chips]');

  const ENDPOINT = '/gift_advisor';

  let selectedOccasion = '';
  let messages = [];

  /* Occasion selection */
  if (chipsEl) {
    chipsEl.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-occasion]');
      if (!btn) return;
      chipsEl.querySelectorAll('.ga-chip').forEach((c) => c.classList.remove('is-active'));
      btn.classList.add('is-active');
      selectedOccasion = btn.dataset.occasion || '';
    });
  }

  /* Auto-resize textarea */
  if (inputEl) {
    inputEl.addEventListener('input', () => {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    });
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        formEl.requestSubmit();
      }
    });
  }

  function setHeroVisibility() {
    if (!heroEl) return;
    heroEl.classList.toggle('is-hidden', messages.length > 0);
  }

  function normalize(text) {
    return String(text || '')
      .replace(/\r\n/g, '\n')
      .replace(/\\n/g, '\n')
      .trim();
  }

  /** Format reply: **bold**, bullets, line breaks. Returns safe HTML. */
  function formatReplyHtml(text) {
    const t = String(text || '').trim();
    if (!t) return '';
    const escaped = t
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
    const withBold = escaped
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    const lines = withBold.split(/\n/).map((l) => l.trim()).filter(Boolean);
    const parts = [];
    let inList = false;
    for (const line of lines) {
      const isBullet = /^[-•*]\s+/.test(line) || /^\d+\.\s+/.test(line);
      const content = line.replace(/^[-•*]\s+|\d+\.\s+/, '');
      if (isBullet) {
        if (!inList) {
          parts.push('<ul>');
          inList = true;
        }
        parts.push('<li>' + content + '</li>');
      } else {
        if (inList) {
          parts.push('</ul>');
          inList = false;
        }
        parts.push('<p>' + line + '</p>');
      }
    }
    if (inList) parts.push('</ul>');
    return parts.join('');
  }

  function stripSearchQueriesFromReply(text) {
    const t = String(text || '').trim();
    const idx = t.search(/\*{0,2}\s*Search\s+Queries\s*\*{0,2}\s*:?\s*/i);
    return idx >= 0 ? t.slice(0, idx).trim() : t;
  }

  function stripJsonReply(text) {
    const t = String(text || '').trim();
    if (!t) return '';
    if (t.startsWith('{') && t.includes('"reply"')) {
      try {
        const obj = JSON.parse(t);
        if (obj && typeof obj.reply === 'string') return obj.reply;
      } catch (_) {}
      const m = t.match(/"reply"\s*:\s*"([\s\S]*?)"(?=\s*[,}])/);
      if (m && m[1]) {
        try {
          return JSON.parse('"' + m[1].replace(/\\/g, '\\\\') + '"');
        } catch (_) {
          return m[1].replace(/\\n/g, '\n').replace(/\\"/g, '"');
        }
      }
    }
    return t;
  }

  function formatQueryTitle(query) {
    return (query || '').replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function renderProductCard(product) {
    const card = document.createElement('a');
    card.className = 'ga-product';
    card.href = product.link || '#';
    card.target = '_blank';
    card.rel = 'noopener noreferrer';

    const imgWrap = document.createElement('div');
    imgWrap.className = 'ga-product__img-wrap';
    const img = document.createElement('div');
    img.className = 'ga-product__img';
    if (product.image) {
      const imgEl = document.createElement('img');
      imgEl.src = product.image;
      imgEl.alt = product.title || 'Product';
      imgEl.loading = 'lazy';
      img.appendChild(imgEl);
    } else {
      img.innerHTML = '<span class="ga-product__placeholder">No image</span>';
    }
    imgWrap.appendChild(img);
    card.appendChild(imgWrap);

    const body = document.createElement('div');
    body.className = 'ga-product__body';
    const title = document.createElement('div');
    title.className = 'ga-product__title';
    title.textContent = product.title || 'Product';
    body.appendChild(title);
    if (product.description) {
      const ratingEl = document.createElement('div');
      ratingEl.className = 'ga-product__rating';
      ratingEl.textContent = product.description.slice(0, 60) + (product.description.length > 60 ? '…' : '');
      body.appendChild(ratingEl);
    }
    const price = document.createElement('div');
    price.className = 'ga-product__price';
    price.textContent = product.price || '';
    body.appendChild(price);
    card.appendChild(body);
    return card;
  }

  function renderMessage(role, text, isTyping = false, productsByQuery = null) {
    const row = document.createElement('div');
    row.className = `ga-msg ga-msg--${role}`;

    const wrap = document.createElement('div');
    wrap.className = 'ga-msg__wrap';

    const bubble = document.createElement('div');
    bubble.className = 'ga-msg__bubble';

    if (isTyping) {
      bubble.innerHTML = '<div class="ga-typing"><span class="ga-typing__dot"></span><span class="ga-typing__dot"></span><span class="ga-typing__dot"></span></div>';
    } else {
      bubble.textContent = normalize(stripJsonReply(text));
      bubble.style.whiteSpace = 'pre-wrap';
    }

    wrap.appendChild(bubble);
    row.appendChild(wrap);

    if (productsByQuery && Array.isArray(productsByQuery) && productsByQuery.length > 0) {
      productsByQuery.forEach(({ query, subtitle, products }) => {
        if (!products || !products.length) return;
        const section = document.createElement('div');
        section.className = 'ga-carousel-section';
        const header = document.createElement('div');
        header.className = 'ga-carousel-section__header';
        const title = document.createElement('h3');
        title.className = 'ga-carousel-section__title';
        title.textContent = formatQueryTitle(query);
        header.appendChild(title);
        if (subtitle) {
          const sub = document.createElement('p');
          sub.className = 'ga-carousel-section__subtitle';
          sub.textContent = subtitle;
          header.appendChild(sub);
        }
        section.appendChild(header);
        const carousel = document.createElement('div');
        carousel.className = 'ga-carousel';
        products.forEach((p) => carousel.appendChild(renderProductCard(p)));
        section.appendChild(carousel);
        wrap.appendChild(section);
      });
    }

    messagesEl.appendChild(row);
    scrollToBottom();
    return { row, bubble };
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function callBackendStream(msg, onDelta, onFinal, onProductsLoading) {
    const history = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-16);

    const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
    const previousQueries = lastAssistant?.products_by_query?.map((p) => p.query).filter(Boolean) || [];
    const payload = {
      message: msg,
      history,
      occasion: selectedOccasion,
      gift_context: lastAssistant?.gift_context || undefined,
      previous_queries: previousQueries.length > 0 ? previousQueries : undefined,
      stream: true,
    };

    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const t = await res.text().catch(() => '');
      let errDetail = t;
      try {
        const j = JSON.parse(t);
        errDetail = j.error || j.message || t;
      } catch (_) {}
      throw new Error(`HTTP ${res.status}: ${String(errDetail).slice(0, 150)}`);
    }
    if (!res.body) {
      throw new Error('No response body (streaming not supported)');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    function parseSseChunk(chunk) {
      buffer += chunk;
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const p of parts) {
        const lines = p.split('\n');
        let event = 'message';
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        if (!data) continue;

        let obj = null;
        try {
          obj = JSON.parse(data);
        } catch (_) {
          obj = null;
        }

        if (event === 'delta' && obj && typeof obj.text === 'string') {
          const clean = String(obj.text || '');
          if (clean) onDelta(clean);
        } else if (event === 'products_loading' && obj && typeof obj === 'object') {
          if (typeof onProductsLoading === 'function') onProductsLoading(obj);
        } else if (event === 'final' && obj && typeof obj === 'object') {
          onFinal(obj);
        }
      }
    }

    let done = false;
    while (!done) {
      const r = await reader.read();
      done = !!r.done;
      const chunk = decoder.decode(r.value || new Uint8Array(), { stream: !done });
      if (chunk) parseSseChunk(chunk);
    }
  }

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = (inputEl.value || '').trim();
    if (!msg) return;

    inputEl.value = '';
    inputEl.style.height = 'auto';

    messages.push({ role: 'user', content: msg });
    renderMessage('user', msg);
    setHeroVisibility();

    sendBtn.disabled = true;

    const { row: typingRow, bubble: typingBubble } = renderMessage('assistant', '', true);
    typingBubble.innerHTML = '<div class="ga-loading"><div class="ga-loading__spinner"></div><span class="ga-loading__text">Finding gifts for you</span><span class="ga-loading__dots"><span></span><span></span><span></span></span></div>';

    try {
      await callBackendStream(
        msg,
        () => {},
        (finalPayload) => {
          const reply = finalPayload.reply || '';
          const productsByQuery = finalPayload.products_by_query || [];
          const replyStr = normalize(stripSearchQueriesFromReply(stripJsonReply(reply)));

          typingBubble.innerHTML = '';

          if (productsByQuery.length > 0) {
            const resultsContainer = document.createElement('div');
            resultsContainer.className = 'ga-results';
            const briefEl = document.createElement('div');
            briefEl.className = 'ga-results__brief';
            briefEl.innerHTML = formatReplyHtml(replyStr);
            resultsContainer.appendChild(briefEl);
            productsByQuery.forEach(({ query, subtitle, products }) => {
              if (!products || !products.length) return;
              const section = document.createElement('div');
              section.className = 'ga-carousel-section';
              const header = document.createElement('div');
              header.className = 'ga-carousel-section__header';
              const title = document.createElement('h3');
              title.className = 'ga-carousel-section__title';
              title.textContent = formatQueryTitle(query);
              header.appendChild(title);
              if (subtitle) {
                const sub = document.createElement('p');
                sub.className = 'ga-carousel-section__subtitle';
                sub.textContent = subtitle;
                header.appendChild(sub);
              }
              section.appendChild(header);
              const carousel = document.createElement('div');
              carousel.className = 'ga-carousel';
              products.forEach((p) => carousel.appendChild(renderProductCard(p)));
              section.appendChild(carousel);
              resultsContainer.appendChild(section);
            });
            typingBubble.appendChild(resultsContainer);
          } else {
            const textEl = document.createElement('div');
            textEl.className = 'ga-reply';
            textEl.innerHTML = formatReplyHtml(replyStr);
            typingBubble.appendChild(textEl);
          }

          messages.push({
            role: 'assistant',
            content: reply,
            products_by_query: productsByQuery,
            gift_context: finalPayload.gift_context,
          });
          scrollToBottom();
        },
        () => {}
      );
    } catch (err) {
      typingBubble.innerHTML = '';
      const errMsg = err.message || 'Unknown error';
      typingBubble.textContent = `Sorry, something went wrong: ${errMsg}`;
      console.error('Gift Advisor error:', err);
      messages.push({ role: 'assistant', content: 'Error: ' + errMsg });
    }

    typingRow.classList.remove('ga-msg--assistant');
    typingRow.classList.add('ga-msg--assistant');
    sendBtn.disabled = false;
    scrollToBottom();
  });
})();
