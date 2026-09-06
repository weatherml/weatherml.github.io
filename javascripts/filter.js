// Client-side filter for the papers page. Uses document-level delegation so it
// keeps working across Material's instant navigation page swaps.
(function () {
  let cards = null;
  let timer = null;

  function buildCache() {
    cards = [];
    document.querySelectorAll(".grid.cards > ul > li").forEach(function (li) {
      cards.push({ el: li, text: li.textContent.toLowerCase() });
    });
    document.querySelectorAll("details.year").forEach(function (d) {
      if (d.dataset.wasOpen === undefined) d.dataset.wasOpen = d.open;
    });
  }

  function apply(query) {
    if (!cards || !cards.length || !cards[0].el.isConnected) buildCache();
    const q = query.trim().toLowerCase();
    const active = q.length >= 2;

    cards.forEach(function (c) {
      c.el.hidden = active && !c.text.includes(q);
    });

    document.querySelectorAll("details.year").forEach(function (d) {
      if (active) {
        const hasMatch = d.querySelector(".grid.cards > ul > li:not([hidden])");
        d.hidden = !hasMatch;
        d.open = !!hasMatch;
      } else {
        d.hidden = false;
        d.open = d.dataset.wasOpen === "true";
      }
    });

    // Hide category headings whose year blocks are all empty
    document.querySelectorAll("article h2").forEach(function (h2) {
      let node = h2.nextElementSibling;
      let visible = false;
      while (node && node.tagName !== "H2") {
        if (node.matches("details.year") && !node.hidden) visible = true;
        node = node.nextElementSibling;
      }
      h2.hidden = active && !visible;
    });
  }

  document.addEventListener("input", function (e) {
    if (e.target.id !== "paper-filter") return;
    clearTimeout(timer);
    timer = setTimeout(function () {
      apply(e.target.value);
    }, 150);
  });
})();
