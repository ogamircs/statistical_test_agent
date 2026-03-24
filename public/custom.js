(function () {
  const HISTORY_PANEL_ID = "conversation-history-panel";
  const HISTORY_LIST_ID = "conversation-history-list";
  const SUGGESTIONS_ID = "starter-suggestions";
  const CONVERSATION_STORAGE_KEY = "ab-testing-agent.conversation-list";
  const ACTIVE_CONVERSATION_KEY = "ab-testing-agent.active-conversation";
  const CLEAR_HISTORY_SUPPRESSION_KEY = "ab-testing-agent.clear-history-suppression";
  const PROCESSING_GIF_URL = "https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif";
  const SUGGESTIONS = [
    "Analyze it",
    "Visualize it",
    "Show me the summary",
    "Compare the segments",
    "Explain the significance",
  ];

  const text = (value) => (value || "").replace(/\s+/g, " ").trim();

  function shell() {
    return document.querySelector("#root > .group\\/sidebar-wrapper > .h-screen.w-screen.flex");
  }

  function appColumn() {
    return shell()?.querySelector(":scope > .flex.flex-col.h-full.w-full") || null;
  }

  function welcomeScreen() {
    return document.querySelector(".welcome-screen");
  }

  function hasMessages() {
    return document.querySelectorAll("[data-step-type]").length > 0;
  }

  function conversationRoot() {
    return document.getElementById("chat-input")?.closest(".flex.w-full.h-full.flex-col.relative") || null;
  }

  function conversationScroller(root = conversationRoot()) {
    return root?.querySelector(":scope > .relative.flex.flex-col.flex-grow.overflow-y-auto") || null;
  }

  function composerShell(root = conversationRoot()) {
    return root?.querySelector(":scope > .flex.flex-col.mx-auto.w-full.p-4.pt-0") || null;
  }

  function clearConversationLayoutClasses() {
    const root = conversationRoot();
    const scroller = conversationScroller(root);
    const composer = composerShell(root);
    root?.classList.remove("centered-conversation-root");
    scroller?.classList.remove("centered-conversation-scroll");
    composer?.classList.remove("centered-conversation-composer");
  }

  function shouldCenterConversation(_messagesPresent) {
    // Never center — keep composer pinned to bottom so content fills the viewport.
    return false;
  }

  function applyConversationLayoutClasses(isCentered) {
    clearConversationLayoutClasses();
    if (!isCentered) {
      return;
    }

    const root = conversationRoot();
    const scroller = conversationScroller(root);
    const composer = composerShell(root);
    root?.classList.add("centered-conversation-root");
    scroller?.classList.add("centered-conversation-scroll");
    composer?.classList.add("centered-conversation-composer");
  }

  function setLayoutState() {
    const messagesPresent = hasMessages();
    const isCenteredConversation = shouldCenterConversation(messagesPresent);
    document.body.classList.add("layout-shell-ready");
    document.body.classList.toggle("layout-empty-state", !messagesPresent && Boolean(welcomeScreen()));
    document.body.classList.toggle("layout-has-messages", messagesPresent);
    document.body.classList.toggle("layout-centered-conversation", isCenteredConversation);
    applyConversationLayoutClasses(isCenteredConversation);
  }

  function clearAllConversations() {
    const activeId = loadActiveConversationId();
    try {
      window.localStorage.removeItem(CONVERSATION_STORAGE_KEY);
      // Keep the active conversation ID so syncConversationList can match it
      // against the suppression marker and avoid re-saving.
      if (activeId) {
        window.sessionStorage.setItem(CLEAR_HISTORY_SUPPRESSION_KEY, activeId);
      }
    } catch (_error) { /* ignore */ }
    const list = document.getElementById(HISTORY_LIST_ID);
    if (list) {
      list.innerHTML =
        '<div class="conversation-history-empty">New conversations will appear here after your first message.</div>';
    }
  }

  function ensureHistoryPanel() {
    const layoutShell = shell();
    const mainColumn = appColumn();
    if (!layoutShell || !mainColumn) return null;

    let panel = document.getElementById(HISTORY_PANEL_ID);
    if (!panel) {
      panel = document.createElement("aside");
      panel.id = HISTORY_PANEL_ID;
      panel.innerHTML = [
        '<div class="history-header-row">',
        '  <div>',
        '    <div class="history-kicker">Navigator</div>',
        '    <div class="history-title">Conversation History</div>',
        '  </div>',
        '  <button type="button" id="clear-history-btn" class="clear-history-btn" title="Clear all conversations">',
        '    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
        '    <span>Clear</span>',
        '  </button>',
        '</div>',
        '<div class="history-caption">One line per conversation in this browser.</div>',
        `<div id="${HISTORY_LIST_ID}"></div>`,
      ].join("");
      panel.querySelector("#clear-history-btn").addEventListener("click", clearAllConversations);
      layoutShell.insertBefore(panel, mainColumn);
    }

    // Ensure clear button exists even if panel was cached from prior session
    if (panel && !panel.querySelector("#clear-history-btn")) {
      const header = panel.querySelector(".history-header-row");
      if (!header) {
        const kicker = panel.querySelector(".history-kicker");
        const title = panel.querySelector(".history-title");
        if (kicker && title) {
          const wrapper = document.createElement("div");
          wrapper.className = "history-header-row";
          const left = document.createElement("div");
          left.appendChild(kicker.cloneNode(true));
          left.appendChild(title.cloneNode(true));
          const btn = document.createElement("button");
          btn.type = "button";
          btn.id = "clear-history-btn";
          btn.className = "clear-history-btn";
          btn.title = "Clear all conversations";
          btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg><span>Clear</span>';
          btn.addEventListener("click", clearAllConversations);
          wrapper.appendChild(left);
          wrapper.appendChild(btn);
          kicker.replaceWith(wrapper);
          title.remove();
        }
      }
    }

    return panel;
  }

  function firstUserMessageTitle() {
    const firstUserStep = document.querySelector('[data-step-type="user_message"]');
    const article = firstUserStep?.querySelector('[role="article"]');
    return text(article?.textContent).slice(0, 120);
  }

  function createConversationId() {
    return `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function persistConversationList(conversations) {
    try {
      window.localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(conversations));
    } catch (_error) {
      // Ignore storage failures; the sidebar can still render from the active DOM state.
    }
  }

  function loadConversationList() {
    try {
      const raw = window.localStorage.getItem(CONVERSATION_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed)
        ? parsed.filter((item) => item && typeof item.id === "string" && typeof item.title === "string")
        : [];
    } catch (_error) {
      return [];
    }
  }

  function loadActiveConversationId() {
    try {
      return window.sessionStorage.getItem(ACTIVE_CONVERSATION_KEY) || "";
    } catch (_error) {
      return "";
    }
  }

  function persistActiveConversationId(conversationId) {
    try {
      window.sessionStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId);
    } catch (_error) {
      // Ignore storage failures; the current chat can still render.
    }
  }

  function clearActiveConversationId() {
    try {
      window.sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    } catch (_error) {
      // Ignore storage failures.
    }
  }

  function loadClearHistorySuppression() {
    try {
      const raw = window.sessionStorage.getItem(CLEAR_HISTORY_SUPPRESSION_KEY);
      return raw || null;
    } catch (_error) {
      return null;
    }
  }

  function clearHistorySuppression() {
    try {
      window.sessionStorage.removeItem(CLEAR_HISTORY_SUPPRESSION_KEY);
    } catch (_error) {
      // Ignore storage failures.
    }
  }

  function syncConversationList() {
    const messagesPresent = hasMessages();
    const title = firstUserMessageTitle();
    const suppressedId = loadClearHistorySuppression();

    if (!messagesPresent || !title) {
      clearActiveConversationId();
      clearHistorySuppression();
      return { conversations: loadConversationList(), activeConversationId: "" };
    }

    const currentConversationId = loadActiveConversationId() || createConversationId();

    // If this conversation was cleared, suppress re-saving for its entire
    // lifetime.  A new conversation (different ID after page reload) will
    // not match and will be tracked normally.
    if (suppressedId && suppressedId === currentConversationId) {
      return { conversations: loadConversationList(), activeConversationId: "" };
    }

    // Suppression for a different (older) conversation – discard it.
    if (suppressedId) {
      clearHistorySuppression();
    }

    const now = new Date().toISOString();
    const previousConversations = loadConversationList();
    const existingConversation = previousConversations.find((item) => item.id === currentConversationId);
    const nextConversation = {
      id: currentConversationId,
      title: existingConversation?.title || title,
      createdAt: existingConversation?.createdAt || now,
      updatedAt: now,
    };
    const remainingConversations = previousConversations.filter((item) => item.id !== currentConversationId);
    const conversations = [nextConversation, ...remainingConversations];

    persistActiveConversationId(currentConversationId);
    persistConversationList(conversations);
    return { conversations, activeConversationId: currentConversationId };
  }

  function renderHistory(conversations, activeConversationId) {
    const panel = ensureHistoryPanel();
    const list = panel?.querySelector(`#${HISTORY_LIST_ID}`);
    if (!list) return;

    const entries = conversations.length ? conversations : loadConversationList();
    if (!entries.length) {
      list.innerHTML =
        '<div class="conversation-history-empty">New conversations will appear here after your first message.</div>';
      return;
    }

    list.innerHTML = "";
    entries.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "conversation-history-item";
      button.classList.toggle("is-active", item.id === activeConversationId);
      button.dataset.conversationId = item.id;
      button.title = item.title;

      const title = document.createElement("div");
      title.className = "conversation-history-title-text";
      title.textContent = item.title;
      button.appendChild(title);

      if (item.id === activeConversationId) {
        button.addEventListener("click", () => {
          conversationScroller()?.scrollTo({ top: 0, behavior: "smooth" });
        });
      }

      list.appendChild(button);
    });
  }

  function buildProcessingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "processing-indicator";

    const image = document.createElement("img");
    image.className = "processing-indicator-gif";
    image.alt = "Loading";
    image.src = PROCESSING_GIF_URL;
    image.decoding = "async";
    image.loading = "eager";
    image.addEventListener("error", () => {
      indicator.classList.add("is-fallback");
    });

    const fallback = document.createElement("span");
    fallback.className = "processing-indicator-fallback";
    fallback.setAttribute("aria-hidden", "true");

    const label = document.createElement("span");
    label.className = "processing-indicator-label";
    label.textContent = "Processing";

    indicator.appendChild(image);
    indicator.appendChild(fallback);
    indicator.appendChild(label);
    return indicator;
  }

  function enhanceProcessingIndicators() {
    document.querySelectorAll('[role="article"]').forEach((article) => {
      const articleText = text(article.textContent);
      const isProcessing = articleText === "Processing...";
      const alreadyEnhanced = article.dataset.processingEnhanced === "true";

      if (isProcessing && !alreadyEnhanced) {
        article.dataset.processingEnhanced = "true";
        article.classList.add("processing-indicator-article");
        article.replaceChildren(buildProcessingIndicator());
        return;
      }

      if (!isProcessing && alreadyEnhanced) {
        delete article.dataset.processingEnhanced;
        article.classList.remove("processing-indicator-article");
      }
    });
  }

  function fillAndSubmitSuggestion(prompt) {
    const input = document.getElementById("chat-input");
    if (!input) return;

    input.focus();
    input.textContent = prompt;
    input.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: prompt }));
    setTimeout(() => {
      const submit = document.getElementById("chat-submit");
      if (submit && !submit.disabled) {
        submit.click();
      }
    }, 40);
  }

  function ensureSuggestions() {
    const screen = welcomeScreen();
    if (!screen) return;

    let container = document.getElementById(SUGGESTIONS_ID);
    if (!container) {
      container = document.createElement("div");
      container.id = SUGGESTIONS_ID;
      screen.appendChild(container);
    }

    if (container.childElementCount > 0) return;

    SUGGESTIONS.forEach((label) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "starter-suggestion";
      button.textContent = label;
      button.addEventListener("click", () => fillAndSubmitSuggestion(label));
      container.appendChild(button);
    });
  }

  function syncConversationChrome() {
    setLayoutState();
    ensureHistoryPanel();
    ensureSuggestions();
    enhanceProcessingIndicators();
    const { conversations, activeConversationId } = syncConversationList();
    renderHistory(conversations, activeConversationId);
  }

  let scheduled = false;
  function scheduleSync() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(() => {
      scheduled = false;
      syncConversationChrome();
    });
  }

  window.addEventListener("load", scheduleSync);
  window.addEventListener("resize", scheduleSync);
  document.addEventListener("readystatechange", scheduleSync);

  const observer = new MutationObserver(scheduleSync);
  observer.observe(document.documentElement, { childList: true, subtree: true });

  scheduleSync();
})();
