import "./styles.css";
import { MASCOT } from "./mascot";

const PIP = "pipx install cozmo-agent";
const VERSION = "0.3.0";
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
  top: HTMLElement;
};

function section(title: string, body: HTMLElement[]): HTMLElement {
  const sec = el("section");
  sec.append(el("h2", undefined, title), ...body);
  return sec;
}

function build(root: HTMLElement): Refs {
  const stage = el("div", "stage");
  const mascot = el("pre", "mascot") as HTMLPreElement;
  mascot.setAttribute("aria-hidden", "true");
  mascot.textContent = MASCOT.join("\n");
  stage.append(mascot);

  const content = el("div", "content");

  const top = el("header", "top");
  const topInner = el("div", "top__inner");
  const meta = el("div", "top__meta");
  meta.innerHTML = `v${VERSION} · <a href="${GITHUB}" target="_blank" rel="noreferrer">github</a>`;
  topInner.append(meta);
  top.append(topInner);

  const hero = el("section", "hero");
  const heroInner = el("div", "hero__inner");
  heroInner.append(
    el("p", "hero__eyebrow", "CLI · multi-model · local-first"),
    el("h1", undefined, "Production-grade agent for your repo"),
    el(
      "p",
      "hero__lead",
      "Orchestrator and worker models, guardrails, session history, and hybrid retrieval - all in one terminal session under your permissions.",
    ),
  );

  const install = el("div", "install");
  const label = el("label");
  label.htmlFor = "install-cmd";
  label.textContent = "Install";
  const row = el("div", "install__row");
  const code = el("code");
  code.id = "install-cmd";
  code.textContent = PIP;
  const copyBtn = el("button", undefined, "Copy") as HTMLButtonElement;
  copyBtn.type = "button";
  row.append(code, copyBtn);
  const hint = el("p", "hint", "");
  install.append(label, row, hint);
  heroInner.append(install);
  hero.append(heroInner);

  const flow = el("section", "flow");
  const flowInner = el("div", "flow__inner");
  flowInner.append(
    el("h2", undefined, "How a turn runs"),
    el(
      "p",
      "flow__lead",
      "Multi-model by role. Soft compaction when context grows. Hard stops on budgets. Subagents explore without polluting the main thread.",
    ),
  );

  const chart = el("div", "flow__chart");
  chart.setAttribute("role", "img");
  chart.setAttribute(
    "aria-label",
    "User to guardrails to orchestrator or worker to tools or subagent to answer, with history on disk",
  );

  const nodes: { id: string; label: string; sub: string }[] = [
    { id: "user", label: "User", sub: "REPL / -m" },
    { id: "rails", label: "Guardrails", sub: "compact · kill" },
    { id: "orch", label: "Orchestrator", sub: "strong model" },
    { id: "work", label: "Worker", sub: "tool loop" },
    { id: "tools", label: "Tools + RAG", sub: "search · edit" },
    { id: "sub", label: "Subagent", sub: "scoped explore" },
    { id: "out", label: "Answer", sub: "history JSONL" },
  ];

  const track = el("div", "flow__track");
  nodes.forEach((n, i) => {
    const cell = el("div", `flow__node flow__node--${n.id}`);
    cell.append(el("span", "flow__label", n.label), el("span", "flow__sub", n.sub));
    track.append(cell);
    if (i < nodes.length - 1) {
      track.append(el("span", "flow__arrow", "→"));
    }
  });
  chart.append(track);

  const split = el("div", "flow__split");
  split.innerHTML =
    '<span class="flow__split-note">worker may call <code>run_subtask</code> → child returns JSON summary only</span>';
  chart.append(split);
  flowInner.append(chart);
  flow.append(flowInner);

  const spec = el("section", "spec");
  const specInner = el("div", "spec__inner");

  specInner.append(
    section("Run", [
      el(
        "p",
        undefined,
        "Open any project and run cozmo. First launch configures provider and model.",
      ),
      (() => {
        const pre = el("pre");
        pre.textContent = "cozmo";
        return pre;
      })(),
    ]),
    section("What it does", [
      (() => {
        const ul = el("ul");
        for (const item of [
          "Multi-model: orchestrator + worker (+ optional verifier)",
          "Guardrails: compact context, max steps, cost/time kills",
          "Subagents for scoped explore without flooding memory",
          "Hybrid RAG + symbols; session history under .cozmo/",
        ]) {
          ul.append(el("li", undefined, item));
        }
        return ul;
      })(),
    ]),
    section("Defaults", [
      el(
        "p",
        undefined,
        "Config in ~/.cozmo/config.json. Repo state in .cozmo/. Shell and writes stay off until you allow them.",
      ),
    ]),
  );

  spec.append(specInner);

  const foot = el("footer", "foot");
  const footInner = el("div", "foot__inner");
  footInner.append(
    el("span", undefined, `cozmo ${VERSION}`),
    (() => {
      const a = el("a", undefined, "github.com/ShashwatXD/cozmo") as HTMLAnchorElement;
      a.href = GITHUB;
      a.target = "_blank";
      a.rel = "noreferrer";
      return a;
    })(),
  );
  foot.append(footInner);

  content.append(top, hero, flow, spec, foot);
  root.append(stage, content);

  return { copyBtn, hint, top };
}

async function copyPip(hint: HTMLElement): Promise<void> {
  try {
    await navigator.clipboard.writeText(PIP);
    hint.textContent = "Copied";
  } catch {
    hint.textContent = "Copy manually";
  }
  window.setTimeout(() => {
    hint.textContent = "";
  }, 1600);
}

function wireScroll(top: HTMLElement): void {
  const onScroll = (): void => {
    top.classList.toggle("is-solid", window.scrollY > 24);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function boot(): void {
  const root = document.querySelector("#app");
  if (!root) return;
  const refs = build(root as HTMLElement);
  refs.copyBtn.addEventListener("click", () => void copyPip(refs.hint));
  wireScroll(refs.top);
}

boot();
