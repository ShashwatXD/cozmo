import "./styles.css";
import { MASCOT } from "./mascot";

const PIP = "pip install cozmo";
const VERSION = "0.2.0";
const GITHUB = "https://github.com/ShashwatXD/cozmo";

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

type Refs = {
  copyBtn: HTMLButtonElement;
  hint: HTMLElement;
  mascot: HTMLPreElement;
  veil: HTMLElement;
  top: HTMLElement;
  splashInner: HTMLElement;
};

function build(root: HTMLElement): Refs {
  const stage = el("div", "stage");
  const mascot = el("pre", "mascot") as HTMLPreElement;
  mascot.setAttribute("aria-hidden", "true");
  mascot.textContent = MASCOT.join("\n");
  const veil = el("div", "stage__veil");
  stage.append(mascot, veil);

  const content = el("div", "content");

  const top = el("div", "top");
  const brand = el("a", "top__brand") as HTMLAnchorElement;
  brand.href = "#";
  brand.setAttribute("aria-label", "cozmo");
  const mark = document.createElement("img");
  mark.className = "top__mark";
  mark.src = "/logo.svg";
  mark.alt = "";
  mark.width = 44;
  mark.height = 22;
  brand.append(mark);
  const meta = el("div", "top__meta");
  meta.innerHTML = `v${VERSION} · <a href="${GITHUB}" target="_blank" rel="noreferrer">source</a>`;
  top.append(brand, meta);

  const splash = el("section", "splash");
  const splashInner = el("div", "splash__inner");
  splashInner.append(
    el("h1", undefined, "ai coding agent"),
    el(
      "p",
      undefined,
      "An LLM that reads your codebase, retrieves relevant context, and uses tools to reason and change code - under your control.",
    ),
  );
  const cmd = el("div", "cmd");
  const code = el("code", undefined, PIP);
  const copyBtn = el("button", undefined, "copy") as HTMLButtonElement;
  copyBtn.type = "button";
  cmd.append(code, copyBtn);
  const hint = el("p", "hint", "");
  splashInner.append(cmd, hint);
  splash.append(splashInner);

  const doc = el("section", "doc");
  const inner = el("div", "doc__inner");

  inner.append(el("h2", undefined, "Models"));
  inner.append(
    el(
      "p",
      undefined,
      "Plug in the model you want. OpenAI, Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint. Same agent loop, swapped LLM behind a single client port. Embeddings follow the same idea (cloud, local, or hash fallback).",
    ),
  );

  inner.append(el("h2", undefined, "Agent"));
  inner.append(
    el(
      "p",
      undefined,
      "A bounded ReAct loop: the model sees memory and tool schemas, decides what to call, gets typed results back, and continues until it answers or hits a step limit. Failures stay structured so the agent can recover instead of dying on the first bad tool call.",
    ),
  );

  inner.append(el("h2", undefined, "Memory of the code"));
  const ul = el("ul");
  for (const item of [
    "Semantic retrieval over embedded chunks (RAG)",
    "Hybrid search — meaning plus exact lexical match",
    "Symbol and reference graphs so the model can follow definitions and callers",
    "Context assembled for the task, not a dump of the whole tree",
  ]) {
    ul.append(el("li", undefined, item));
  }
  inner.append(ul);

  inner.append(el("h2", undefined, "Tools"));
  inner.append(
    el(
      "p",
      undefined,
      "The model isn’t limited to chat. It can search, read, write, inspect git, and run gated commands. Permissions decide what is allowed; the workspace sandbox keeps paths inside the project.",
    ),
  );

  inner.append(el("h2", undefined, "Ports"));
  inner.append(
    el(
      "p",
      undefined,
      "Domain code talks to LLMClient and Embedder interfaces — not vendor SDKs. Adapters live in infra. That keeps the agent logic stable while models and providers change.",
    ),
  );
  const arch = el("pre");
  arch.textContent = [
    "agent loop  →  LLMClient port  →  OpenAI | Anthropic | Ollama | …",
    "retrieval    →  Embedder port   →  OpenAI | Ollama | hash",
    "tools        →  registry        →  search · read · write · git · shell",
  ].join("\n");
  inner.append(arch);

  inner.append(el("h2", undefined, "Cost & safety"));
  inner.append(
    el(
      "p",
      undefined,
      "Token usage and estimated cost are tracked per session. Writes and shell are permission-gated. Paths are sandboxed. API keys are never logged.",
    ),
  );

  inner.append(el("h2", undefined, "Install"));
  inner.append(
    el(
      "p",
      undefined,
      "Python 3.11+. First run asks for provider and model, then builds a local index of the repo so the agent can retrieve context.",
    ),
  );
  const runPre = el("pre");
  runPre.textContent = ["pip install cozmo", "cozmo"].join("\n");
  inner.append(runPre);

  doc.append(inner);

  const foot = el("div", "foot");
  foot.append(
    el("span", undefined, `cozmo ${VERSION}`),
    (() => {
      const a = el("a", undefined, GITHUB.replace("https://", "")) as HTMLAnchorElement;
      a.href = GITHUB;
      a.target = "_blank";
      a.rel = "noreferrer";
      return a;
    })(),
  );

  content.append(top, splash, doc, foot);
  root.append(stage, content);

  return { copyBtn, hint, mascot, veil, top, splashInner };
}

async function copyPip(hint: HTMLElement): Promise<void> {
  try {
    await navigator.clipboard.writeText(PIP);
    hint.textContent = "copied";
  } catch {
    hint.textContent = "select and copy manually";
  }
  window.setTimeout(() => {
    hint.textContent = "";
  }, 1600);
}

function wireScroll(refs: Refs): void {
  const { mascot, veil, top, splashInner } = refs;
  let ticking = false;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const ease = (t: number): number => t * t * (3 - 2 * t);

  const update = (): void => {
    ticking = false;
    const y = window.scrollY;
    const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const p = ease(Math.min(1, y / max));
    const heroH = Math.max(1, window.innerHeight * 0.9);
    const heroP = ease(Math.min(1, y / heroH));

    const scale = 1 + heroP * 0.05 + p * 0.03;
    const ty = y * 0.04;
    mascot.style.transform = `translate3d(0, ${ty.toFixed(2)}px, 0) scale(${scale.toFixed(4)})`;
    mascot.style.opacity = String(Math.max(0.4, 0.72 - heroP * 0.16 - p * 0.1));
    veil.style.opacity = String(heroP * 0.25 + p * 0.15);
    top.classList.toggle("is-solid", y > 32);

    splashInner.style.opacity = String(Math.max(0.4, 1 - heroP * 0.45));
    splashInner.style.transform = `translate3d(0, ${(heroP * -10).toFixed(2)}px, 0)`;
  };

  const onScroll = (): void => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(update);
    }
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  update();
}

function boot(): void {
  const root = document.querySelector("#app");
  if (!root) return;
  const refs = build(root as HTMLElement);
  refs.copyBtn.addEventListener("click", () => void copyPip(refs.hint));
  wireScroll(refs);
}

boot();
