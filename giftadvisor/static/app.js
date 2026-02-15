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
    const payload = {
      message: msg,
      history,
      occasion: selectedOccasion,
      gift_context: lastAssistant && lastAssistant.gift_context ? lastAssistant.gift_context : undefined,
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

    let fullText = '';

    let productsLoadingEl = null;

    try {
      await callBackendStream(
        msg,
        (delta) => {
          fullText += delta;
          typingBubble.innerHTML = '';
          typingBubble.textContent = normalize(stripJsonReply(fullText));
          typingBubble.style.whiteSpace = 'pre-wrap';
          scrollToBottom();
        },
        (finalPayload) => {
          const reply = finalPayload.reply || fullText;
          const productsByQuery = finalPayload.products_by_query || [];
          const wrap = typingRow.querySelector('.ga-msg__wrap');
          if (productsLoadingEl && productsLoadingEl.parentNode) {
            productsLoadingEl.remove();
          }
          if (productsByQuery.length > 0) {
            const resultsContainer = document.createElement('div');
            resultsContainer.className = 'ga-results';
            const briefText = document.createElement('div');
            briefText.className = 'ga-results__brief';
            const replyStr = normalize(stripJsonReply(reply));
            briefText.textContent = replyStr.length > 200 ? replyStr.slice(0, 200).trim() + '…' : replyStr;
            briefText.style.whiteSpace = 'pre-wrap';
            resultsContainer.appendChild(briefText);
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
            typingBubble.innerHTML = '';
            typingBubble.appendChild(resultsContainer);
          } else {
            typingBubble.innerHTML = '';
            typingBubble.textContent = normalize(stripJsonReply(reply));
            typingBubble.style.whiteSpace = 'pre-wrap';
          }
          typingRow.classList.remove('ga-msg--assistant');
          typingRow.classList.add('ga-msg--assistant');
          messages.push({
            role: 'assistant',
            content: reply,
            products_by_query: productsByQuery,
            gift_context: finalPayload.gift_context,
          });
        },
        (loadingPayload) => {
          const wrap = typingRow.querySelector('.ga-msg__wrap');
          if (!wrap) return;
          productsLoadingEl = document.createElement('div');
          productsLoadingEl.className = 'ga-products-loading';
          productsLoadingEl.innerHTML = '<div class="ga-products-loading__spinner"></div><span class="ga-products-loading__text">Finding products...</span>';
          wrap.appendChild(productsLoadingEl);
          scrollToBottom();
        }
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
