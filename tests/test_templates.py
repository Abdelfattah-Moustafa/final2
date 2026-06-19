"""Frontend template regression tests.

These guard the invariants that broke (and were fixed) during the Phase 6
redesign, so the same classes of bug can't silently return:

  * the dashboards must keep the {% block head %} that loads the Emscripten
    WASM shim, MediaPipe and socket.io (dropping it killed camera + meetings);
  * the design-system stylesheets / theme controller must stay linked;
  * the legacy CSS hooks the preserved inline JS still queries must exist
    (.module-panel, .meet-entry-row, .meet-stage);
  * every getElementById(...) the inline JS performs must resolve to a real
    element id in the markup (a missing one threw on the Enter key);
  * the Arabic dashboard must be RTL, force the ArSL model, and carry no
    leftover English UI labels;
  * the inline JS must parse (node --check) so an edit can't ship a syntax error.

They are intentionally dependency-light: templates are rendered with Jinja2
directly (the same way FastAPI's Jinja2Templates does) so the suite needs no
database, ML models, or network.
"""
import os
import re
import shutil
import subprocess

import pytest
from jinja2 import Environment, FileSystemLoader

TPL_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "templates")
_env = Environment(loader=FileSystemLoader(TPL_DIR), autoescape=True)

ALL_TEMPLATES = ["landing.html", "login.html", "signup.html", "index.html", "index_ar.html"]
DASHBOARDS = ["index.html", "index_ar.html"]

# Ids the inline JS references but that legitimately have no markup element:
#   debug-error-banner*  – created on demand by showDebugError()
#   topbar-signout       – optional duplicate of the sidebar sign-out (guarded)
#   tts-btn              – orphan reference; toggleTTS is never wired to a button
SAFE_MISSING_IDS = {"debug-error-banner", "debug-error-banner-text", "topbar-signout", "tts-btn"}


class _StubRequest:
    """FastAPI injects `request`; templates here use static paths, but provide
    a no-op url_for just in case a future template starts using it."""
    def url_for(self, *args, **kwargs):
        return ""


def render(name):
    return _env.get_template(name).render(request=_StubRequest())


def split_markup_and_script(html):
    """Return (markup, inline_js) for a dashboard template.

    base.html appends its own <script> (the nav-populate IIFE) after the
    content block, so we take only the FIRST inline block after app.js — the
    dashboard's own JS — up to its matching </script>.
    """
    marker = '<script src="/static/js/app.js"></script>'
    assert marker in html
    markup, _, rest = html.partition(marker)
    inline_js = rest.partition("<script>")[2].partition("</script>")[0]
    return markup, inline_js


# ── Every template renders ────────────────────────────────────
@pytest.mark.parametrize("tpl", ALL_TEMPLATES)
def test_template_renders(tpl):
    html = render(tpl)
    assert html.strip(), f"{tpl} rendered empty"


# ── Dashboards keep their runtime libraries (dropped-head regression) ──
@pytest.mark.parametrize("tpl", DASHBOARDS)
def test_dashboard_loads_runtime_libraries(tpl):
    html = render(tpl)
    for needle in (
        "holistic.js",
        "camera_utils",
        "drawing_utils",
        "/static/js/socket.io.js",
        "Object.defineProperty",   # Emscripten getter shim
        "/static/js/app.js",
    ):
        assert needle in html, f"{tpl} is missing required runtime: {needle}"


# ── Dashboards link the design system ─────────────────────────
@pytest.mark.parametrize("tpl", DASHBOARDS)
def test_dashboard_links_design_system(tpl):
    html = render(tpl)
    for needle in ("/static/css/theme.css", "/static/css/ui.css", "/static/js/theme.js"):
        assert needle in html, f"{tpl} no longer links {needle}"


# ── Legacy hooks the inline JS still queries ──────────────────
@pytest.mark.parametrize("tpl", DASHBOARDS)
def test_legacy_js_hooks_present(tpl):
    html = render(tpl)
    assert html.count('class="d-panel module-panel') == 6, "expected 6 module panels"
    assert "meet-entry-row" in html
    assert "meet-stage" in html
    assert "meeting-avatar-corner" in html


# ── Every getElementById target exists in the markup ──────────
@pytest.mark.parametrize("tpl", DASHBOARDS)
def test_js_dom_ids_resolve(tpl):
    html = render(tpl)
    markup, inline_js = split_markup_and_script(html)
    markup_ids = set(re.findall(r'id="([\w-]+)"', html))
    referenced = set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", inline_js))
    missing = referenced - markup_ids - SAFE_MISSING_IDS
    assert not missing, f"{tpl}: inline JS references missing element ids: {sorted(missing)}"


# ── Arabic dashboard: RTL + ArSL + no English leaks ───────────
def test_arabic_dashboard_is_rtl_and_arabic():
    html = render("index_ar.html")
    assert 'dir="rtl"' in html
    assert 'lang="ar"' in html
    assert "forcedLang = 'ar'" in html


def test_arabic_dashboard_has_no_english_label_leaks():
    html = render("index_ar.html")
    markup, _ = split_markup_and_script(html)
    leaks = [
        s for s in (
            ">Sign to Text<", ">Sign to Speech<", ">Text to Sign<", ">Speech to Sign<",
            ">Live Meeting<", ">Start Camera<", ">Session Log<", ">Not connected<",
            ">Create Meeting<", ">Participants<",
        )
        if s in markup
    ]
    assert not leaks, f"English UI labels leaked into the Arabic dashboard markup: {leaks}"


# ── Inline JS parses (catches an edit shipping a syntax error) ──
@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize("tpl", DASHBOARDS)
def test_inline_js_syntax_valid(tpl, tmp_path):
    _, inline_js = split_markup_and_script(render(tpl))
    js_file = tmp_path / f"{tpl}.js"
    js_file.write_text(inline_js, encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(js_file)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"{tpl} inline JS syntax error:\n{result.stderr}"
