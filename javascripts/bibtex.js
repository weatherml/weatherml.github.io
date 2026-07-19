document.addEventListener("click", function (e) {
  const link = e.target.closest("a.bibtex-link");
  if (link) {
    e.preventDefault();
    fetch(link.href)
      .then((r) => r.text())
      .then((text) => {
        navigator.clipboard.writeText(text);
        const original = link.innerHTML;
        link.textContent = "Copied!";
        setTimeout(() => (link.innerHTML = original), 1500);
      });
    return;
  }

  const toggle = e.target.closest(".abstract-toggle");
  if (toggle) {
    const id = toggle.dataset.id;
    const snippet = document.getElementById("snip-" + id);
    const full = document.getElementById("full-" + id);
    const card = toggle.closest("li");
    if (full.hidden) {
      snippet.hidden = true;
      full.hidden = false;
      toggle.textContent = "less";
      if (card) card.classList.add("card-expanded");
    } else {
      snippet.hidden = false;
      full.hidden = true;
      toggle.textContent = "more";
      if (card) card.classList.remove("card-expanded");
    }
  }
});
