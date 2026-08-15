import "./styles.css";

const GITHUB = "https://github.com/ShashwatXD/cozmo";
const VERSION = "0.3.0";

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function link(href: string, label: string, className?: string): HTMLAnchorElement {
  const a = el("a", className, label) as HTMLAnchorElement;
  a.href = href;
  if (href.startsWith("http")) {
    a.target = "_blank";
    a.rel = "noreferrer";
  }
  return a;
}

function buildNav(active: "home" | "docs"): HTMLElement {
  const top = el("header", "top is-solid");
  const inner = el("div", "top__inner top__inner--nav");
  const brand = link("./", "cozmo", "top__brand");
  const nav = el("nav", "top__nav");
  nav.append(
    link("./", "Home", active === "home" ? "is-active" : undefined),
    link("./docs.html", "How it works", active === "docs" ? "is-active" : undefined),
    link(GITHUB, "GitHub"),
  );
  const meta = el("div", "top__meta");
  meta.textContent = `v${VERSION}`;
  inner.append(brand, nav, meta);
  top.append(inner);
  return top;
}

type Topic = {
  id: string;
  title: string;
  blurb: string;
  body: HTMLElement;
};

function topicPanel(topic: Topic, accordion: HTMLElement): HTMLElement {
  const item = el("div", "docs-acc__item");
  item.dataset.id = topic.id;

  const btn = el("button", "docs-acc__btn") as HTMLButtonElement;
  btn.type = "button";
  btn.setAttribute("aria-expanded", "false");
  btn.setAttribute("aria-controls", `docs-panel-${topic.id}`);

  const title = el("span", "docs-acc__title", topic.title);
  const blurb = el("span", "docs-acc__blurb", topic.blurb);
  const chevron = el("span", "docs-acc__chev", "+");
  chevron.setAttribute("aria-hidden", "true");
  btn.append(title, blurb, chevron);

  const panel = el("div", "docs-acc__panel");
  panel.id = `docs-panel-${topic.id}`;
  panel.hidden = true;
  panel.append(topic.body);

  btn.addEventListener("click", () => {
    const opening = panel.hidden;
    for (const other of accordion.querySelectorAll<HTMLElement>(".docs-acc__item")) {
      const otherBtn = other.querySelector<HTMLButtonElement>(".docs-acc__btn");
      const otherPanel = other.querySelector<HTMLElement>(".docs-acc__panel");
      const otherChev = other.querySelector(".docs-acc__chev");
      if (!otherBtn || !otherPanel) continue;
      const isThis = other === item;
      const open = isThis && opening;
      otherPanel.hidden = !open;
      otherBtn.setAttribute("aria-expanded", open ? "true" : "false");
      other.classList.toggle("is-open", open);
      if (otherChev) otherChev.textContent = open ? "−" : "+";
    }
  });

  item.append(btn, panel);
  return item;
}

function para(...lines: string[]): HTMLElement {
  const wrap = el("div", "docs-acc__copy");
  for (const line of lines) {
    wrap.append(el("p", undefined, line));
  }
  return wrap;
}

function turnBody(): HTMLElement {
  const wrap = el("div", "docs-acc__copy");
  wrap.append(
    el(
      "p",
      undefined,
      "User input hits guardrails. The worker model runs the tool loop. The orchestrator handles compaction (and can be a stronger model). Subagents explore in a tight budget and return a compact evidence pack, not a full chat dump.",
    ),
  );
  const chart = el("div", "flow__chart docs-chart");
  const track = el("div", "flow__track");
  const nodes = [
    { label: "User", sub: "REPL / -m" },
    { label: "Guardrails", sub: "compact · kill" },
    { label: "Worker", sub: "ReAct tools" },
    { label: "Evidence", sub: "search · read" },
    { label: "Answer", sub: "cited · metered" },
  ];
  nodes.forEach((n, i) => {
    const cell = el("div", "flow__node");
    cell.append(el("span", "flow__label", n.label), el("span", "flow__sub", n.sub));
    track.append(cell);
    if (i < nodes.length - 1) track.append(el("span", "flow__arrow", "→"));
  });
  chart.append(track);
  wrap.append(chart);
  return wrap;
}

function archBody(): HTMLElement {
  const wrap = el("div", "docs-acc__copy");
  wrap.append(
    el(
      "p",
      undefined,
      "Fixed layering: CLI presentation, application use-cases, pure domain, and swappable infra. Search and RAG sit beside the agent, not tangled into the prompt stack.",
    ),
  );
  const layers = el("div", "docs-layers");
  const rows: [string, string][] = [
    ["cli/", "REPL, modes, permission prompts, context meter"],
    ["app/", "AgentRunner, model router, compaction, sessions"],
    ["domain/", "Messages, policies, ports. No SDK imports."],
    ["infra/", "LLM providers, tools, RAG store, history JSONL"],
  ];
  for (const [path, desc] of rows) {
    const row = el("div", "docs-layer");
    row.append(el("code", undefined, path), el("span", undefined, desc));
    layers.append(row);
  }
  wrap.append(layers);
  return wrap;
}

function craftBody(): HTMLElement {
  const wrap = el("div", "docs-acc__copy");
  const list = el("ul", "docs-list");
  for (const item of [
    "Multi-model: orchestrator for compaction, optional cheaper worker_model for the tool loop",
    "BYOK multi-provider: OpenAI, Anthropic, OpenRouter, Ollama, OpenAI-compatible endpoints",
    "Offline test suite with stub LLM/embedder. CI does not need API keys.",
    "Session continue and export; investigation trails under .cozmo/history/",
    "Incremental RAG indexing with graceful degrade when embeddings fail",
    "Typed tool registry, result size caps, and workspace path sandboxing",
  ]) {
    list.append(el("li", undefined, item));
  }
  wrap.append(list);
  return wrap;
}

function build(root: HTMLElement): void {
  const page = el("div", "content docs-page");

  const hero = el("section", "docs-hero");
  const heroInner = el("div", "docs-hero__inner");
  heroInner.append(
    el("p", "hero__eyebrow", "Documentation · architecture · craft"),
    el("h1", undefined, "How Cozmo works"),
    el(
      "p",
      "hero__lead",
      "A local-first CLI coding agent with an owned ReAct loop: multi-model routing, hybrid retrieval, and permission-gated tools, without a LangChain core.",
    ),
    el("p", "docs-hint", "Click a topic to expand."),
  );
  hero.append(heroInner);

  const topicsSec = el("section", "docs-topics");
  const shell = el("div", "docs-shell");
  shell.append(el("h2", "docs-kicker", "Topics"));

  const accordion = el("div", "docs-acc");
  const topics: Topic[] = [
    {
      id: "runtime",
      title: "Owned runtime",
      blurb: "You control the loop",
      body: para(
        "You own steps, tools, memory, and stop reasons. Domain ports stay free of provider SDKs: clean layers that are easy to reason about and test.",
      ),
    },
    {
      id: "models",
      title: "Multi-model routing",
      blurb: "Orchestrator + worker",
      body: para(
        "Roles, not a single chat model. The worker runs the ReAct tool loop. The orchestrator handles compaction and heavier synthesis when you route a stronger model there.",
        "Set model for the orchestrator and an optional cheaper worker_model for tool calls. Same provider or mixed. BYOK across OpenAI, Anthropic, OpenRouter, and Ollama.",
      ),
    },
    {
      id: "retrieval",
      title: "Hybrid retrieval",
      blurb: "Exact find + meaning",
      body: para(
        "Exact find via ripgrep-backed search. Meaning via BM25 + embeddings, lexical rerank, then narrow file reads. Citations stay grounded in path:line.",
      ),
    },
    {
      id: "trust",
      title: "Trust by default",
      blurb: "Modes + permission prompts",
      body: para(
        "Ask / Plan / Agent modes. Writes and shell need preview approval (once / always / deny). Workspace sandbox keeps tools inside the repo.",
      ),
    },
    {
      id: "context",
      title: "Context economy",
      blurb: "Budgets you can see",
      body: para(
        "Soft compaction, hard budgets on steps, tools, cost, and time. Live ctx≈N/budget meter so sessions stay observable, not opaque.",
      ),
    },
    {
      id: "turn",
      title: "A single turn",
      blurb: "Guardrails → tools → answer",
      body: turnBody(),
    },
    {
      id: "arch",
      title: "Architecture",
      blurb: "cli → app → domain ← infra",
      body: archBody(),
    },
    {
      id: "craft",
      title: "Built like a product",
      blurb: "Tests, sessions, BYOK",
      body: craftBody(),
    },
  ];

  for (const topic of topics) {
    accordion.append(topicPanel(topic, accordion));
  }
  shell.append(accordion);
  topicsSec.append(shell);

  const stack = el("section", "docs-block docs-block--last");
  const stackInner = el("div", "docs-shell");
  stackInner.append(
    el("h2", "docs-kicker", "Stack"),
    el(
      "p",
      "docs-stack",
      "Python 3.11+ · Typer · pytest · hybrid BM25 + vectors · optional Chroma · MIT",
    ),
    (() => {
      const actions = el("div", "docs-actions");
      actions.append(
        link(GITHUB, "View source →", "docs-cta"),
        link("./", "Back to install", "docs-cta docs-cta--ghost"),
      );
      return actions;
    })(),
  );
  stack.append(stackInner);

  const foot = el("footer", "foot");
  const footInner = el("div", "foot__inner");
  footInner.append(
    el("span", undefined, `cozmo ${VERSION}`),
    link(GITHUB, "github.com/ShashwatXD/cozmo"),
  );
  foot.append(footInner);

  page.append(buildNav("docs"), hero, topicsSec, stack, foot);
  root.append(page);
}

function boot(): void {
  const root = document.querySelector("#app");
  if (!root) return;
  build(root as HTMLElement);
}

boot();
