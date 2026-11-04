/**
 * The guide as a popup over the designer: `help/index.html` (also a page of its own) read once and shown in a
 * <dialog>, its links resolved against that page, its contents links scrolling within the popup.
 */
const GUIDE_URL = new URL('../help/index.html', import.meta.url);

let dialog = null;

async function build() {
  const res = await fetch(GUIDE_URL);
  if (!res.ok) throw new Error(`guide not found (${res.status})`);
  const page = new DOMParser().parseFromString(await res.text(), 'text/html');
  const main = page.querySelector('main');
  if (!main) throw new Error('guide has no content');
  // Links point where they would from the guide's own page; "back to the designer" is this page, so it goes.
  for (const a of main.querySelectorAll('a[href]')) {
    const href = a.getAttribute('href');
    if (href.startsWith('#')) continue;
    const url = new URL(href, GUIDE_URL);
    if (url.pathname === new URL('../', GUIDE_URL).pathname) { a.closest('p')?.remove(); continue; }
    a.href = url.href;
    a.target = '_blank';
    a.rel = 'noopener';
  }
  // Images too: their paths are the guide page's.
  for (const img of main.querySelectorAll('img[src]')) img.src = new URL(img.getAttribute('src'), GUIDE_URL).href;
  const d = document.createElement('dialog');
  d.className = 'rfd4p-guide';
  d.setAttribute('aria-label', 'Guide');
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'rfd4p-guide-x';
  close.textContent = '×';
  close.title = 'Close';
  close.setAttribute('aria-label', 'Close the guide');
  close.addEventListener('click', () => d.close());
  const body = document.createElement('div');
  body.className = 'rfd4p-guide-body';
  body.append(...main.childNodes);
  // The contents scroll the popup rather than the page behind it.
  body.addEventListener('click', (ev) => {
    const a = ev.target.closest?.('a[href^="#"]');
    if (!a) return;
    ev.preventDefault();
    body.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  // A click on the backdrop closes it.
  d.addEventListener('click', (ev) => { if (ev.target === d) d.close(); });
  d.append(close, body);
  document.body.append(d);
  return d;
}

/** Open the guide; offline or if it cannot be read, the guide's own page instead. */
export async function openGuide() {
  try {
    dialog ??= await build();
    dialog.showModal();
  } catch (err) {
    console.warn('Guide popup unavailable:', err);
    window.open(GUIDE_URL.href, '_blank', 'noopener');
  }
}
