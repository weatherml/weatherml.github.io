document.addEventListener("click", function (e) {
  const link = e.target.closest("a.bibtex-link");
  if (!link) return;
  e.preventDefault();
  fetch(link.href)
    .then((r) => r.text())
    .then((text) => {
      navigator.clipboard.writeText(text);
      const original = link.innerHTML;
      link.textContent = "Copied!";
      setTimeout(() => (link.innerHTML = original), 1500);
    });
});
