import "./styles.css";
import { MASCOT } from "./mascot";

const PIP = "pipx install cozmo-agent";
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
  meta.innerHTML = `v${VERSION} · <a href="${GITHUB}" target="_blank" rel="noreferrer">github</a>`;
  top.append(brand, meta);

  const hero = el("section", "hero");
  const heroInner = el("div", "hero__inner");
  heroInner.append(
    el("h1", undefined, "Coding agent for your repo"),
    el(
      "p",
      "hero__lead",
      "Runs locally. Indexes the project once, then answers and edits with tools under your permissions.",
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

  const spec = el("section", "spec");
  const specInner = el("div", "spec__inner");

  specInner.append(
    section("Run", [
      el("p", undefined, "After install, open any project and run cozmo. First launch configures provider and model."),
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
          "ReAct loop with bounded steps and structured tool errors",
          "Hybrid retrieval: embeddings plus lexical search over the index",
          "Symbol and reference lookup from a parsed code index",
          "OpenAI, OpenRouter, Ollama, or any OpenAI-compatible API",
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
        "Config lives in ~/.cozmo/config.json. Repo state in .cozmo/. Shell and writes are off unless you allow them.",
      ),
    ]),
  );

  spec.append(specInner);

  const foot = el("footer", "foot");
  foot.append(
    el("span", undefined, `cozmo ${VERSION}`),
    (() => {
      const a = el("a", undefined, "github.com/ShashwatXD/cozmo") as HTMLAnchorElement;
      a.href = GITHUB;
      a.target = "_blank";
      a.rel = "noreferrer";
      return a;
    })(),
  );

  content.append(top, hero, spec, foot);
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
