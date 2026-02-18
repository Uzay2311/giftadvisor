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
  const chatEl = app.querySelector('.ga-chat');
  const heroEl = app.querySelector('[data-ga-hero]');
  const occasionEl = app.querySelector('[data-ga-occasion]');
  const chipsEl = app.querySelector('[data-ga-chips]');
  const budgetChipsEl = app.querySelector('[data-ga-budget-chips]');
  let feedbackToastTimer = null;

  const ENDPOINT = '/gift_advisor';
  const DEVICE_ID_KEY = 'giftadvisor_device_id_v1';
  const INITIAL_ASSISTANT_MESSAGES = [
    "Let's explore a great gift together.",
    "Let's find a gift they'll truly love.",
    "Tell me who you're shopping for, and we'll find something special.",
    "Let's discover a thoughtful gift for your loved one.",
    "Ready to explore gift ideas? Let's start with your loved one.",
    "Let's pick a meaningful gift that feels just right.",
  ];

  let selectedOccasion = '';
  let selectedBudget = { min: null, max: null };
  let messages = [];
  let activeProfileId = '';
  let peopleProfiles = null;
  const likedProductsByProfile = new Map();
  const dislikedProductsByProfile = new Map();

  function createLocalDeviceId() {
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
      }
    } catch (_) {}
    const rand = Math.random().toString(36).slice(2, 12);
    return 'dev-' + Date.now().toString(36) + '-' + rand;
  }

  function getOrCreateDeviceId() {
    try {
      const existing = localStorage.getItem(DEVICE_ID_KEY);
      if (existing && /^[a-z0-9._-]{8,128}$/i.test(existing)) return existing.toLowerCase();
      const created = createLocalDeviceId().toLowerCase().replace(/[^a-z0-9._-]/g, '-').slice(0, 128);
      localStorage.setItem(DEVICE_ID_KEY, created);
      return created;
    } catch (_) {
      return createLocalDeviceId().toLowerCase().replace(/[^a-z0-9._-]/g, '-').slice(0, 128);
    }
  }

  const deviceId = getOrCreateDeviceId();
  // Do not persist chip selections; clear any legacy stored values.
  try {
    localStorage.removeItem('giftadvisor_selected_occasion_v1');
    localStorage.removeItem('giftadvisor_selected_budget_v1');
  } catch (_) {}

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

  /* Budget selection */
  if (budgetChipsEl) {
    budgetChipsEl.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-budget-min],[data-budget-max]');
      if (!btn) return;
      budgetChipsEl.querySelectorAll('.ga-chip').forEach((c) => c.classList.remove('is-active'));
      btn.classList.add('is-active');
      const minRaw = btn.dataset.budgetMin;
      const maxRaw = btn.dataset.budgetMax;
      const min = Number.parseInt(minRaw || '', 10);
      const max = Number.parseInt(maxRaw || '', 10);
      selectedBudget = {
        min: Number.isFinite(min) ? min : null,
        max: Number.isFinite(max) ? max : null,
      };
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

  function getInitialAssistantMessage() {
    if (!Array.isArray(INITIAL_ASSISTANT_MESSAGES) || INITIAL_ASSISTANT_MESSAGES.length === 0) {
      return "Let's explore a great gift together.";
    }
    const idx = Math.floor(Math.random() * INITIAL_ASSISTANT_MESSAGES.length);
    return INITIAL_ASSISTANT_MESSAGES[idx];
  }

  async function typeAssistantText(el, text, scrollMode = 'page') {
    if (!el) return;
    const full = normalize(stripSearchQueriesFromReply(stripJsonReply(text)));
    el.textContent = full || '';
    if (scrollMode === 'messages' && messagesEl) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else {
      scrollToBottom(true);
    }
  }

  function renderInitialAssistantMessage() {
    if (!messagesEl) return;
    if (heroEl) heroEl.classList.add('is-hidden');
    if (chatEl) chatEl.classList.add('ga-chat--intro-compact');
    messagesEl.classList.add('ga-messages--intro');
    const { bubble } = renderMessage('assistant', '');
    bubble.innerHTML = '';
    const textEl = document.createElement('div');
    textEl.className = 'ga-reply';
    textEl.style.whiteSpace = 'pre-wrap';
    bubble.appendChild(textEl);
    typeAssistantText(textEl, getInitialAssistantMessage(), 'messages').then(() => {
      textEl.innerHTML = formatReplyHtml(textEl.textContent || '');
    });
    if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
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

  function formatReviews(num) {
    if (num == null || num === '') return '';
    const n = parseInt(num, 10);
    if (isNaN(n)) return String(num);
    return n.toLocaleString();
  }

  function renderStars(rating) {
    const r = parseFloat(rating);
    if (isNaN(r) || r < 0) return '';
    const full = Math.floor(r);
    const partial = r - full;
    const empty = 5 - full - (partial > 0 ? 1 : 0);
    let html = '';
    for (let i = 0; i < full; i++) html += '<span class="ga-star ga-star--full"></span>';
    if (partial > 0) html += '<span class="ga-star ga-star--partial" style="--fill:' + (partial * 100) + '%"></span>';
    for (let i = 0; i < empty; i++) html += '<span class="ga-star ga-star--empty"></span>';
    return html;
  }

  function _feedbackProfileKey() {
    return String(activeProfileId || 'default').trim() || 'default';
  }

  function _normalizeFeedbackProduct(product) {
    const title = String((product && product.title) || '').replace(/\s+/g, ' ').trim();
    const price = String((product && product.price) || '').trim();
    if (!title) return null;
    return { title, price };
  }

  function _feedbackTitleKey(title) {
    return String(title || '').replace(/\s+/g, ' ').trim().toLowerCase();
  }

  function _getFeedbackList(mapObj, key) {
    const arr = mapObj.get(key);
    return Array.isArray(arr) ? arr : [];
  }

  function _setFeedbackList(mapObj, key, arr) {
    mapObj.set(key, Array.isArray(arr) ? arr.slice(0, 30) : []);
  }

  function _getCurrentFeedbackForProduct(product) {
    const p = _normalizeFeedbackProduct(product);
    if (!p) return { liked: false, disliked: false };
    const key = _feedbackProfileKey();
    const tkey = _feedbackTitleKey(p.title);
    const liked = _getFeedbackList(likedProductsByProfile, key).some((x) => _feedbackTitleKey(x.title) === tkey);
    const disliked = _getFeedbackList(dislikedProductsByProfile, key).some((x) => _feedbackTitleKey(x.title) === tkey);
    return { liked, disliked };
  }

  function _toggleProductFeedback(product, reaction) {
    const p = _normalizeFeedbackProduct(product);
    if (!p) return { liked: false, disliked: false };
    const key = _feedbackProfileKey();
    const tkey = _feedbackTitleKey(p.title);
    let liked = _getFeedbackList(likedProductsByProfile, key).filter((x) => _feedbackTitleKey(x.title) !== tkey);
    let disliked = _getFeedbackList(dislikedProductsByProfile, key).filter((x) => _feedbackTitleKey(x.title) !== tkey);
    const current = _getCurrentFeedbackForProduct(p);
    if (reaction === 'like') {
      if (!current.liked) liked.unshift(p);
    } else if (reaction === 'dislike') {
      if (!current.disliked) disliked.unshift(p);
    }
    _setFeedbackList(likedProductsByProfile, key, liked);
    _setFeedbackList(dislikedProductsByProfile, key, disliked);
    return _getCurrentFeedbackForProduct(p);
  }

  function _getFeedbackPayloadForActiveProfile() {
    const key = _feedbackProfileKey();
    return {
      liked_products: _getFeedbackList(likedProductsByProfile, key).slice(0, 20),
      disliked_products: _getFeedbackList(dislikedProductsByProfile, key).slice(0, 20),
    };
  }

  function showFeedbackToast(text) {
    if (!app) return;
    let toast = app.querySelector('[data-ga-feedback-toast]');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'ga-feedback-toast';
      toast.setAttribute('data-ga-feedback-toast', '1');
      toast.setAttribute('aria-live', 'polite');
      app.appendChild(toast);
    }
    toast.textContent = String(text || '').trim() || 'Saved preference';
    toast.classList.add('is-visible');
    if (feedbackToastTimer) clearTimeout(feedbackToastTimer);
    feedbackToastTimer = setTimeout(() => {
      toast.classList.remove('is-visible');
    }, 950);
  }

  function renderProductCard(product) {
    const card = document.createElement('div');
    card.className = 'ga-product';
    const link = document.createElement('a');
    link.className = 'ga-product__link';
    link.href = product.link || '#';
    link.target = '_blank';
    link.rel = 'noopener noreferrer';

    const imgWrap = document.createElement('div');
    imgWrap.className = 'ga-product__img-wrap';
    const img = document.createElement('div');
    img.className = 'ga-product__img';
    if (product.image) {
      const imgEl = document.createElement('img');
      imgEl.src = product.image;
      imgEl.alt = product.title || 'Product';
      imgEl.loading = 'lazy';
      imgEl.addEventListener('load', () => scrollToBottom());
      imgEl.addEventListener('error', () => scrollToBottom());
      img.appendChild(imgEl);
    } else {
      img.innerHTML = '<span class="ga-product__placeholder">No image</span>';
    }
    imgWrap.appendChild(img);
    link.appendChild(imgWrap);

    const body = document.createElement('div');
    body.className = 'ga-product__body';
    const title = document.createElement('div');
    title.className = 'ga-product__title';
    title.textContent = product.title || 'Product';
    body.appendChild(title);
    if (product.rating != null || product.reviews != null) {
      const ratingRow = document.createElement('div');
      ratingRow.className = 'ga-product__rating';
      if (product.rating != null) {
        const starsWrap = document.createElement('span');
        starsWrap.className = 'ga-product__stars';
        starsWrap.innerHTML = renderStars(product.rating);
        ratingRow.appendChild(starsWrap);
      }
      const ratingText = document.createElement('span');
      ratingText.className = 'ga-product__rating-text';
      const parts = [];
      if (product.rating != null) parts.push(String(product.rating));
      if (product.reviews != null) parts.push(formatReviews(product.reviews) + ' reviews');
      ratingText.textContent = parts.join(' · ');
      ratingRow.appendChild(ratingText);
      body.appendChild(ratingRow);
    }
    const price = document.createElement('div');
    price.className = 'ga-product__price';
    price.textContent = product.price || '';
    body.appendChild(price);
    link.appendChild(body);
    card.appendChild(link);

    const actions = document.createElement('div');
    actions.className = 'ga-product__actions';
    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'ga-feedback ga-feedback--up';
    upBtn.setAttribute('aria-label', 'Like this product');
    upBtn.title = 'Like this';
    upBtn.textContent = '👍';
    const downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.className = 'ga-feedback ga-feedback--down';
    downBtn.setAttribute('aria-label', 'Dislike this product');
    downBtn.title = 'Dislike this';
    downBtn.textContent = '👎';

    function applyFeedbackUI() {
      const state = _getCurrentFeedbackForProduct(product);
      upBtn.classList.toggle('is-active', !!state.liked);
      downBtn.classList.toggle('is-active', !!state.disliked);
      upBtn.setAttribute('aria-pressed', state.liked ? 'true' : 'false');
      downBtn.setAttribute('aria-pressed', state.disliked ? 'true' : 'false');
    }
    upBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const before = _getCurrentFeedbackForProduct(product);
      _toggleProductFeedback(product, 'like');
      const after = _getCurrentFeedbackForProduct(product);
      applyFeedbackUI();
      if (after.liked && !before.liked) showFeedbackToast('Liked');
      else if (!after.liked && before.liked) showFeedbackToast('Like removed');
      else if (after.liked && before.disliked) showFeedbackToast('Switched to like');
    });
    downBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const before = _getCurrentFeedbackForProduct(product);
      _toggleProductFeedback(product, 'dislike');
      const after = _getCurrentFeedbackForProduct(product);
      applyFeedbackUI();
      if (after.disliked && !before.disliked) showFeedbackToast('Disliked');
      else if (!after.disliked && before.disliked) showFeedbackToast('Dislike removed');
      else if (after.disliked && before.liked) showFeedbackToast('Switched to dislike');
    });
    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    card.appendChild(actions);
    applyFeedbackUI();
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
        title.textContent = (subtitle && subtitle.trim()) ? subtitle.trim() : formatQueryTitle(query);
        header.appendChild(title);
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

  function scrollToBottom(force = false) {
    if (!messagesEl) return;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    // Keep page anchored to latest content always.
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'auto' });
    if (formEl) {
      const rect = formEl.getBoundingClientRect();
      const isOutOfView = rect.bottom > window.innerHeight || rect.top < 0;
      if (isOutOfView) {
        formEl.scrollIntoView({ block: 'end', inline: 'nearest', behavior: 'auto' });
      }
    }
  }

  function scheduleAutoScroll(force = false) {
    scrollToBottom(force);
    requestAnimationFrame(() => scrollToBottom(force));
    setTimeout(() => scrollToBottom(force), 120);
  }

  function getAccumulatedContext() {
    const assistants = messages.filter((m) => m.role === 'assistant');
    let giftContext = {};
    let latestActiveProfileId = activeProfileId || '';
    let latestPeopleProfiles = peopleProfiles;
    const previousProductsByProfile = new Map();
    for (const a of assistants) {
      if (a.people_profiles && typeof a.people_profiles === 'object') {
        latestPeopleProfiles = a.people_profiles;
      }
      if (a.active_profile_id && typeof a.active_profile_id === 'string') {
        latestActiveProfileId = a.active_profile_id;
      }
      if (!latestPeopleProfiles && a.gift_context && typeof a.gift_context === 'object') {
        giftContext = { ...giftContext, ...a.gift_context };
      }
      if (a.products_by_query && Array.isArray(a.products_by_query)) {
        const pid = String(a.active_profile_id || '').trim();
        if (!pid) continue;
        previousProductsByProfile.set(pid, a.products_by_query);
      }
    }
    if (latestPeopleProfiles && latestActiveProfileId) {
      const profiles = Array.isArray(latestPeopleProfiles.profiles) ? latestPeopleProfiles.profiles : [];
      const active = profiles.find((p) => p && p.id === latestActiveProfileId);
      if (active && active.context && typeof active.context === 'object') {
        giftContext = { ...active.context };
      }
    }
    const previousProductsByQuery = previousProductsByProfile.get(latestActiveProfileId) || [];
    const previousQueries = previousProductsByQuery.map((p) => p.query).filter(Boolean);
    if (selectedOccasion) {
      giftContext.occasion = selectedOccasion;
    }
    if (Number.isFinite(selectedBudget.min)) {
      giftContext.budget_min = selectedBudget.min;
    }
    if (Number.isFinite(selectedBudget.max)) {
      giftContext.budget_max = selectedBudget.max;
    }
    const feedback = _getFeedbackPayloadForActiveProfile();
    if (feedback.liked_products.length > 0) {
      giftContext.liked_products = feedback.liked_products;
    }
    if (feedback.disliked_products.length > 0) {
      giftContext.disliked_products = feedback.disliked_products;
    }
    return {
      gift_context: giftContext,
      previous_queries: previousQueries,
      previous_products_by_query: previousProductsByQuery,
      people_profiles: latestPeopleProfiles || undefined,
      active_profile_id: latestActiveProfileId || undefined,
      liked_products: feedback.liked_products,
      disliked_products: feedback.disliked_products,
    };
  }

  async function callBackendStream(msg, onDelta, onFinal, onProductsLoading) {
    const history = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-16);

    const { gift_context, previous_queries, previous_products_by_query, people_profiles, active_profile_id, liked_products, disliked_products } = getAccumulatedContext();
    const payload = {
      message: msg,
      history,
      occasion: selectedOccasion,
      budget_min: Number.isFinite(selectedBudget.min) ? selectedBudget.min : undefined,
      budget_max: Number.isFinite(selectedBudget.max) ? selectedBudget.max : undefined,
      device_id: deviceId,
      gift_context: Object.keys(gift_context).length > 0 ? gift_context : undefined,
      people_profiles,
      active_profile_id,
      previous_queries: previous_queries.length > 0 ? previous_queries : undefined,
      previous_products_by_query: previous_products_by_query.length > 0 ? previous_products_by_query : undefined,
      liked_products: liked_products.length > 0 ? liked_products : undefined,
      disliked_products: disliked_products.length > 0 ? disliked_products : undefined,
      stream: true,
    };

    const controller = new AbortController();
    const STREAM_TIMEOUT_MS = 70000;
    const timeoutId = setTimeout(() => {
      try { controller.abort(); } catch (_) {}
    }, STREAM_TIMEOUT_MS);

    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
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
    let finalReceived = false;
    let doneReceived = false;

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
          finalReceived = true;
          onFinal(obj);
        } else if (event === 'done') {
          doneReceived = true;
        } else if (event === 'error') {
          const msg = (obj && (obj.error || obj.message)) || 'Server error';
          throw new Error(String(msg));
        }
      }
    }

    let done = false;
    try {
      while (!done) {
        const r = await reader.read();
        done = !!r.done;
        const chunk = decoder.decode(r.value || new Uint8Array(), { stream: !done });
        if (chunk) parseSseChunk(chunk);
      }
    } finally {
      clearTimeout(timeoutId);
    }
    if (!finalReceived) {
      if (doneReceived) {
        throw new Error('No final payload received from stream');
      }
      throw new Error('Stream ended before final response');
    }
  }

  renderInitialAssistantMessage();

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = (inputEl.value || '').trim();
    if (!msg) return;
    if (chatEl) chatEl.classList.remove('ga-chat--intro-compact');
    messagesEl.classList.remove('ga-messages--intro');

    inputEl.value = '';
    inputEl.style.height = 'auto';

    messages.push({ role: 'user', content: msg });
    renderMessage('user', msg);
    scheduleAutoScroll(true);
    setHeroVisibility();

    sendBtn.disabled = true;

    const { row: typingRow, bubble: typingBubble } = renderMessage('assistant', '', true);
    const renderLoadingState = (text) => {
      const safeText = String(text || '').trim();
      if (!safeText) {
        typingBubble.innerHTML = '<div class="ga-loading"><div class="ga-loading__spinner"></div></div>';
        return;
      }
      typingBubble.innerHTML =
        '<div class="ga-loading">' +
        '<div class="ga-loading__spinner"></div>' +
        '<span class="ga-loading__text">' + safeText + '</span>' +
        '</div>';
    };
    renderLoadingState('');

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
            const gotOptionsEl = document.createElement('div');
            gotOptionsEl.className = 'ga-reply';
            gotOptionsEl.textContent = 'Got some options';
            resultsContainer.appendChild(gotOptionsEl);
            const briefEl = document.createElement('div');
            briefEl.className = 'ga-results__brief';
            briefEl.style.whiteSpace = 'pre-wrap';
            resultsContainer.appendChild(briefEl);
            productsByQuery.forEach(({ query, subtitle, products }) => {
              if (!products || !products.length) return;
              const section = document.createElement('div');
              section.className = 'ga-carousel-section';
              const header = document.createElement('div');
              header.className = 'ga-carousel-section__header';
              const title = document.createElement('h3');
              title.className = 'ga-carousel-section__title';
              title.textContent = (subtitle && subtitle.trim()) ? subtitle.trim() : formatQueryTitle(query);
              header.appendChild(title);
              section.appendChild(header);
              const carousel = document.createElement('div');
              carousel.className = 'ga-carousel';
              products.forEach((p) => carousel.appendChild(renderProductCard(p)));
              section.appendChild(carousel);
              resultsContainer.appendChild(section);
            });
            typingBubble.appendChild(resultsContainer);
            typeAssistantText(briefEl, replyStr).then(() => {
              briefEl.innerHTML = formatReplyHtml(replyStr);
            });
          } else {
            const textEl = document.createElement('div');
            textEl.className = 'ga-reply';
            textEl.style.whiteSpace = 'pre-wrap';
            typingBubble.appendChild(textEl);
            typeAssistantText(textEl, replyStr).then(() => {
              textEl.innerHTML = formatReplyHtml(replyStr);
            });
          }

          messages.push({
            role: 'assistant',
            content: reply,
            products_by_query: productsByQuery,
            gift_context: finalPayload.gift_context,
            people_profiles: finalPayload.people_profiles,
            active_profile_id: finalPayload.active_profile_id,
          });
          if (finalPayload.active_profile_id) activeProfileId = finalPayload.active_profile_id;
          if (finalPayload.people_profiles && typeof finalPayload.people_profiles === 'object') {
            peopleProfiles = finalPayload.people_profiles;
          }
          scheduleAutoScroll();
        },
        (loadingPayload) => {
          if (loadingPayload && loadingPayload.queries && loadingPayload.queries.length > 0) {
            const qCount = loadingPayload.queries.length;
            const msg = qCount > 1
              ? 'Searching best options'
              : 'Choosing the best options';
            renderLoadingState(msg);
            scheduleAutoScroll();
          }
        }
      );
    } catch (err) {
      typingBubble.innerHTML = '';
      let errMsg = err && err.message ? String(err.message) : 'Unknown error';
      if (err && (err.name === 'AbortError' || /aborted|timed out/i.test(errMsg))) {
        errMsg = 'Request timed out. Please try again.';
      }
      typingBubble.textContent = `Sorry, something went wrong: ${errMsg}`;
      console.error('Gift Advisor error:', err);
      messages.push({ role: 'assistant', content: 'Error: ' + errMsg });
    }

    typingRow.classList.remove('ga-msg--assistant');
    typingRow.classList.add('ga-msg--assistant');
    sendBtn.disabled = false;
    scheduleAutoScroll(true);
  });
})();
