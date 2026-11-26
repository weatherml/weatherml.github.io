// Tag filter on the Explore page: toggle chips to show the papers that have
// every selected tag (all papers when none are selected). The selection
// lives in the URL (?t=a,b) so it can be shared, and chips under the result
// cards add their tag to the selection.
(function () {
  const PAGE = 30;
  const ICONS = {
    paper: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M6 2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm0 2h7v5h5v11H6zm2 8v2h8v-2zm0 4v2h5v-2z"/></svg>',
    pdf: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2m-9.5 8.5c0 .8-.7 1.5-1.5 1.5H7v2H5.5V9H8c.8 0 1.5.7 1.5 1.5zm5 2c0 .8-.7 1.5-1.5 1.5h-2.5V9H13c.8 0 1.5.7 1.5 1.5zm4-3H17v1h1.5V13H17v2h-1.5V9h3zm-6.5 0h1v3h-1zm-5 0h1v1H7z"/></svg>',
    copy: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 21H8V7h11m0-2H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2m-3-4H4a2 2 0 0 0-2 2v14h2V3h12z"/></svg>',
    github: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="currentColor" d="M216.5 362.5c-66-8-112.5-55.5-112.5-117 0-25 9-52 24-70-6.5-16.5-5.5-51.5 2-66 20-2.5 47 8 63 22.5 19-6 39-9 63.5-9s44.5 3 62.5 8.5c15.5-14 43-24.5 63-22 7 13.5 8 48.5 1.5 65.5 16 19 24.5 44.5 24.5 70.5 0 61.5-46.5 108-113.5 116.5 17 11 28.5 35 28.5 62.5v52c0 15 12.5 23.5 27.5 17.5C441 459.5 512 369 512 257 512 115.5 397 0 255.5 0S0 115.5 0 257c0 111 70.5 203 165.5 237.5 13.5 5 26.5-4 26.5-17.5v-40c-7 3-16 5-24 5-33 0-52.5-18-66.5-51.5-5.5-13.5-11.5-21.5-23-23-6-.5-8-3-8-6 0-6 10-10.5 20-10.5 14.5 0 27 9 40 27.5 10 14.5 20.5 21 33 21s20.5-4.5 32-16c8.5-8.5 15-16 21-21"/></svg>',
  };

  let data = null; // papers.json, cached across page swaps

  function icon(name) {
    return '<span class="twemoji">' + ICONS[name] + "</span>";
  }

  function card(p) {
    const links = [
      '<a href="https://arxiv.org/abs/' + p.v + '">' + icon("paper") + " arXiv</a>",
      '<a href="https://arxiv.org/pdf/' + p.v + '">' + icon("pdf") + " PDF</a>",
    ];
    if (p.g) links.push('<a href="' + p.g + '">' + icon("github") + " Code</a>");
    links.push('<a class="bibtex-link" href="../bibtex/' + p.id + '.bib">' + icon("copy") + " BibTeX</a>");
    const chips = p.k
      .map(function (slug) {
        return '<a class="md-tag" href="?t=' + slug + '" data-tag="' + slug + '">' + data.tags[slug] + "</a>";
      })
      .join(" ");
    return (
      "<li>" +
      "<h4>" + p.t + "</h4>" +
      '<p class="paper-meta"><em>' + p.a + "</em> · " + p.m + "</p>" +
      "<p>" + p.s + ' <a class="abstract-toggle" href="../' + p.u + "#" + p.id + '">more</a></p>' +
      '<p class="paper-links">' + links.join(" · ") + "</p>" +
      (chips ? '<p class="paper-tags">' + chips + "</p>" : "") +
      "</li>"
    );
  }

  function setup(root) {
    const chips = Array.from(root.querySelectorAll(".tag-chip"));
    const status = root.querySelector(".tag-status");
    const results = root.querySelector("#tag-results");
    const more = root.querySelector("#tag-more");
    const params = new URLSearchParams(location.search);
    const selected = new Set((params.get("t") || "").split(",").filter(Boolean));
    let matches = [];
    let shown = 0;

    function keys(p) {
      return p.keys || (p.keys = new Set(p.k.concat([p.c])));
    }

    function showMore() {
      const next = matches.slice(shown, shown + PAGE);
      const ul = results.querySelector("ul");
      ul.insertAdjacentHTML("beforeend", next.map(card).join(""));
      if (window.renderMath) window.renderMath(ul);
      shown += next.length;
      more.hidden = shown >= matches.length;
    }

    function update() {
      const tags = Array.from(selected);
      matches = data.papers.filter(function (p) {
        return tags.every(function (t) { return keys(p).has(t); });
      });

      // Each chip shows how many papers you'd get by adding it
      chips.forEach(function (chip) {
        const slug = chip.dataset.tag;
        const on = selected.has(slug);
        const n = on ? matches.length : matches.filter(function (p) { return keys(p).has(slug); }).length;
        chip.classList.toggle("is-selected", on);
        chip.disabled = !on && n === 0;
        chip.querySelector(".tag-chip-count").textContent = n;
      });

      history.replaceState(null, "", tags.length ? "?t=" + tags.join(",") : location.pathname);

      // With nothing selected every paper matches, so the page opens on the
      // newest papers
      results.innerHTML = "<ul></ul>";
      shown = 0;
      status.innerHTML =
        matches.length + (matches.length === 1 ? " paper" : " papers") +
        (tags.length ? ' · <a href="#" class="tag-clear">Clear selection</a>' : "");
      showMore();
    }

    root.addEventListener("click", function (e) {
      const chip = e.target.closest(".tag-chip");
      const cardTag = e.target.closest("#tag-results a.md-tag");
      if (chip) {
        const slug = chip.dataset.tag;
        selected.has(slug) ? selected.delete(slug) : selected.add(slug);
        update();
      } else if (cardTag) {
        // Keep instant navigation from following the link as well
        e.preventDefault();
        e.stopPropagation();
        selected.add(cardTag.dataset.tag);
        update();
        root.scrollIntoView({ behavior: "smooth" });
      } else if (e.target.closest(".tag-clear")) {
        e.preventDefault();
        e.stopPropagation();
        selected.clear();
        update();
      } else if (e.target === more) {
        showMore();
      }
    });

    update();
  }

  document$.subscribe(function () {
    const root = document.getElementById("tag-filter");
    if (!root) return;
    const ready = data
      ? Promise.resolve()
      : fetch(root.dataset.src).then(function (r) { return r.json(); }).then(function (d) { data = d; });
    ready.then(function () { setup(root); });
  });
})();
