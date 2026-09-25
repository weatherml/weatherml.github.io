// Render the math that pymdownx.arithmatex wrapped in .arithmatex elements.
// Runs on every page load, including Material's instant navigation swaps;
// tag-filter.js also calls renderMath() on the cards it adds.
window.renderMath = function (root) {
  root.querySelectorAll(".arithmatex").forEach(function (el) {
    const tex = el.textContent.trim();
    // arithmatex emits \( ... \) for inline and \[ ... \] for display math
    katex.render(tex.slice(2, -2), el, {
      displayMode: tex.startsWith("\\["),
      throwOnError: false,
      errorColor: "inherit",
    });
  });
};

document$.subscribe(function () {
  window.renderMath(document);
});
