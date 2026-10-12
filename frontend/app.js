"use strict";

// Generation can be slow (free LLM, catalog rate limits); stop waiting after this.
const REQUEST_TIMEOUT_MS = 180_000;

const $ = (id) => document.getElementById(id);
const form = $("style-form");
const description = $("description");
const limitPrice = $("limit-price");
const maxPriceField = $("max-price-field");
const maxPrice = $("max-price");
const freeOnly = $("free-only");
const generateButton = $("generate");
const message = $("message");
const loading = $("loading");
const loadingSeconds = $("loading-seconds");
const results = $("results");
const resultsHeading = $("results-heading");
const resultsNote = $("results-note");
const slotsContainer = $("slots");
const copyButton = $("copy-links");

let links = [];
let loadingTimer;
let copyTimer;

class HttpError extends Error {
  constructor(status, body) {
    super(`HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

limitPrice.addEventListener("change", () => {
  maxPriceField.hidden = !limitPrice.checked;
  maxPrice.disabled = !limitPrice.checked;
  if (limitPrice.checked) maxPrice.focus();
});

for (const field of [description, maxPrice]) {
  field.addEventListener("input", () => field.removeAttribute("aria-invalid"));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const request = readForm();
  if (request) generate(request);
});

copyButton.addEventListener("click", async () => {
  const copied = await copyText(links.join("\n"));
  copyButton.textContent = copied ? `Copied ${plural(links.length, "link")}` : "Couldn't copy";
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => (copyButton.textContent = "Copy all links"), 2000);
});

/** The /generate request body, or null (with a message shown) when the form is invalid. */
function readForm() {
  hideMessage();
  const text = description.value.trim();
  if (!text) {
    return invalid(description, "Describe the style you're going for first.");
  }
  let max = null;
  if (limitPrice.checked) {
    max = Number(maxPrice.value);
    if (maxPrice.value === "" || !Number.isInteger(max) || max < 0) {
      return invalid(maxPrice, "Enter a max price as a whole number of Robux.");
    }
  }
  return {
    description: text,
    price_filter_enabled: limitPrice.checked,
    max_price: max,
    free_only: freeOnly.checked,
  };
}

function invalid(field, text) {
  field.setAttribute("aria-invalid", "true");
  field.focus();
  showMessage(text);
  return null;
}

async function generate(request) {
  setLoading(true);
  results.hidden = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let slots;
  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new HttpError(response.status, await response.text());
    }
    slots = (await response.json()).slots;
  } catch (error) {
    console.error("Outfit generation failed:", error, error.body ?? "");
    showMessage(friendlyError(error));
    return;
  } finally {
    clearTimeout(timeout);
    setLoading(false);
  }
  showResults(Array.isArray(slots) ? slots : []);
}

function friendlyError(error) {
  if (error.name === "AbortError") {
    return "This is taking longer than usual. Please try again in a moment.";
  }
  if (error instanceof HttpError) {
    if (error.status === 422) {
      return "The stylist couldn't read that request. Check the description and price, then try again.";
    }
    if (error.status === 502) {
      return "The Roblox catalog isn't responding right now. Please try again in a minute.";
    }
    if ([404, 405, 501].includes(error.status)) {
      // A static file server (e.g. VS Code's Live Server) served this page and has no /generate.
      return "This page isn't connected to the stylist backend. Open it from the FastAPI server instead, e.g. http://127.0.0.1:8000/";
    }
    return "Something went wrong while styling your outfit. Please try again.";
  }
  if (error instanceof TypeError) {
    return "Couldn't reach the stylist. Check your connection (and that the server is running), then try again.";
  }
  return "Something went wrong while styling your outfit. Please try again.";
}

function setLoading(on) {
  generateButton.disabled = on;
  generateButton.textContent = on ? "Generating…" : "Generate outfit";
  loading.hidden = !on;
  clearInterval(loadingTimer);
  if (on) {
    const started = Date.now();
    const tick = () => (loadingSeconds.textContent = `${Math.floor((Date.now() - started) / 1000)}s`);
    tick();
    loadingTimer = setInterval(tick, 1000);
  }
}

function showResults(slots) {
  slotsContainer.replaceChildren(...slots.map(renderSlot));
  links = slots.flatMap((slot) => (slot.items ?? []).map((item) => item.roblox_url));
  copyButton.disabled = links.length === 0;
  resultsNote.hidden = links.length > 0;
  results.hidden = false;
  resultsHeading.focus({ preventScroll: true });
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  results.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

function renderSlot(slot) {
  const items = slot.items ?? [];
  return el("details", { className: "slot", open: true }, [
    el("summary", {}, [
      el("span", { className: "slot-name", textContent: slot.slot }),
      slot.keyword ? el("span", { className: "slot-keyword", textContent: slot.keyword }) : null,
      el("span", { className: "slot-count", textContent: plural(items.length, "item") }),
    ]),
    items.length
      ? el("div", { className: "cards" }, items.map(renderItem))
      : el("p", { className: "empty", textContent: "Nothing matched for this slot." }),
  ]);
}

function renderItem(item) {
  const price = formatPrice(item.price);
  return el("a", { className: "card", href: item.roblox_url, target: "_blank", rel: "noopener noreferrer" }, [
    thumbnail(item.thumbnail_url),
    el("span", { className: "card-name", textContent: item.name || "Unnamed item" }),
    el("span", { className: `card-price ${price.kind}`, textContent: price.text }),
  ]);
}

function thumbnail(url) {
  const missing = () => el("span", { className: "thumb thumb-missing", textContent: "No preview" });
  if (!url) return missing();
  const image = el("img", { className: "thumb", src: url, alt: "", loading: "lazy", width: 420, height: 420 });
  image.addEventListener("error", () => image.replaceWith(missing()), { once: true });
  return image;
}

/** The backend sends 0 for free items and null for items that can't be bought right now. */
function formatPrice(price) {
  if (price === 0) return { text: "Free", kind: "free" };
  if (price === null || price === undefined) return { text: "Off sale", kind: "off-sale" };
  return { text: `R$ ${price.toLocaleString()}`, kind: "" };
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // navigator.clipboard only exists on https/localhost (not e.g. a phone on the LAN).
    const scratch = el("textarea", { value: text, readOnly: true });
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.append(scratch);
    scratch.select();
    const copied = document.execCommand("copy");
    scratch.remove();
    return copied;
  }
}

function showMessage(text) {
  message.textContent = text;
  message.hidden = false;
}

function hideMessage() {
  message.hidden = true;
  message.textContent = "";
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter(Boolean));
  return node;
}
