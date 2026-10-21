// Copy buttons, share link, contents rail, reading progress, and reveal motion.

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

async function copyText(text, label, done) {
  const before = label.textContent;
  try {
    await navigator.clipboard.writeText(text);
    label.textContent = done;
  } catch {
    label.textContent = "Copy failed";
  }
  setTimeout(() => { label.textContent = before; }, 1600);
}

document.addEventListener("click", (event) => {
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    const block = copy.closest(".code");
    // With line numbers, Chroma renders a table; the code is in the last cell.
    const source = block.querySelector("td:last-child pre code") || block.querySelector("pre code") || block.querySelector("pre");
    const clone = source.cloneNode(true);
    clone.querySelectorAll(".ln").forEach((n) => n.remove());
    copyText(clone.textContent.replace(/\n$/, ""), copy, "Copied");
    return;
  }
  const share = event.target.closest("[data-share]");
  if (share) {
    copyText(share.dataset.share, share.querySelector("span") || share, "Link copied");
  }
});

// Mark the section being read in the contents rail.
const rail = document.querySelector(".toc--rail");
if (rail && "IntersectionObserver" in window) {
  const links = new Map();
  rail.querySelectorAll('a[href^="#"]').forEach((a) => {
    links.set(decodeURIComponent(a.hash.slice(1)), a);
  });
  const headings = [...document.querySelectorAll(".prose h2[id], .prose h3[id]")].filter((h) => links.has(h.id));
  let current = null;
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        if (current) current.removeAttribute("aria-current");
        current = links.get(entry.target.id);
        current.setAttribute("aria-current", "true");
      }
    }
  }, { rootMargin: "0px 0px -70% 0px" });
  headings.forEach((h) => observer.observe(h));
}

// Reading progress: a hairline under the header that fills as the post is read.
const progress = document.querySelector(".progress");
const article = document.querySelector(".prose--post");
if (progress && article) {
  let ticking = false;
  const update = () => {
    const r = article.getBoundingClientRect();
    const total = r.height - window.innerHeight * 0.6;
    const p = Math.min(1, Math.max(0, -r.top / Math.max(1, total)));
    progress.style.transform = `scaleX(${p})`;
    ticking = false;
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

// Reveal motion, played once when an element first scrolls into view:
// the featured tile pattern sweeps in, chart bars grow, stat values count up.
if (!reduceMotion && "IntersectionObserver" in window) {
  document.documentElement.classList.add("motion-ready");
  // Numbers below the fold start from zero; ones already on screen stay put.
  document.querySelectorAll("[data-count]").forEach((n) => {
    if (n.getBoundingClientRect().top > innerHeight) zero(n);
    else n.removeAttribute("data-count");
  });
  const play = (node) => {
    if (node.matches(".mask--animate")) node.classList.add("is-playing");
    if (node.matches(".chart")) node.classList.add("is-playing");
    if (node.matches("[data-count]")) countUp(node);
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { play(e.target); io.unobserve(e.target); }
    }
  }, { threshold: 0.25 });
  document.querySelectorAll(".mask--animate, .chart, [data-count]").forEach((n) => io.observe(n));
} else {
  document.documentElement.classList.remove("js");
}

function parts(node) {
  if (!node.dataset.final) node.dataset.final = node.textContent;
  return node.dataset.final.match(/^([^\d]*)([\d.]+)(.*)$/);
}
function zero(node) {
  const m = parts(node);
  if (m) node.textContent = m[1] + (0).toFixed((m[2].split(".")[1] || "").length) + m[3];
}
function countUp(node) {
  const m = parts(node);
  if (!m) return;
  const [, pre, num, post] = m;
  const target = parseFloat(num), digits = (num.split(".")[1] || "").length;
  const t0 = performance.now(), dur = 900;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur);
    const eased = 1 - (1 - k) ** 3;
    node.textContent = pre + (target * eased).toFixed(digits) + post;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
