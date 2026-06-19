/* ============================================================
   TOGETHER — Theme controller (light / dark)
   ------------------------------------------------------------
   Load in <head> WITHOUT defer so the first block runs before
   paint (no flash). Persists the choice; falls back to the OS
   preference. Exposes window.Theme for pages/components.
   ============================================================ */
(function () {
  var KEY = 'together-theme';
  var root = document.documentElement;

  function systemPref() {
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
  }
  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function resolve() {
    var s = stored();
    return (s === 'light' || s === 'dark') ? s : systemPref();
  }

  // ── First-paint: apply immediately (this runs synchronously) ──
  var current = resolve();
  root.setAttribute('data-theme', current);

  function apply(theme, persist) {
    current = theme;
    root.setAttribute('data-theme', theme);
    if (persist) { try { localStorage.setItem(KEY, theme); } catch (e) {} }
    syncControls();
    document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
  }

  function syncControls() {
    // Toggle buttons: <div class="t-toggle"><button data-theme-set="light">..</button>..</div>
    document.querySelectorAll('[data-theme-set]').forEach(function (btn) {
      btn.setAttribute('aria-pressed', btn.getAttribute('data-theme-set') === current ? 'true' : 'false');
    });
  }

  window.Theme = {
    get: function () { return current; },
    set: function (t) { apply(t === 'light' ? 'light' : 'dark', true); },
    toggle: function () { apply(current === 'dark' ? 'light' : 'dark', true); }
  };

  // ── Wire controls + follow OS changes when no explicit choice ──
  function init() {
    syncControls();
    document.querySelectorAll('[data-theme-set]').forEach(function (btn) {
      btn.addEventListener('click', function () { window.Theme.set(btn.getAttribute('data-theme-set')); });
    });
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      btn.addEventListener('click', function () { window.Theme.toggle(); });
    });
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function (e) {
        if (!stored()) apply(e.matches ? 'light' : 'dark', false);
      });
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
