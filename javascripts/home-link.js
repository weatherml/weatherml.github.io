// Make the site name in the header go home, like the logo next to it.
// Delegated so it survives Material's instant navigation page swaps.
document.addEventListener("click", function (e) {
  if (!e.target.closest(".md-header__topic:first-child")) return;
  const logo = document.querySelector(".md-header__button.md-logo");
  if (logo) logo.click();
});
