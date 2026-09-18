"""Generate the FLAMINGO pipeline diagram (landscape, for a slide).

    uv run python images/make_pipeline_diagram.py        # writes flamingo_pipeline.svg
    # then render the PNG (see the bottom of this file for the Chrome command)

Palette and semantics follow the repository's own figures: pink is the causal
pathway and what travels between sites, coral is what differs per site plus the
confounding, and a dashed edge marks a benchmark that needs pooled individual
rows. 16:9 so it drops straight onto a slide.
"""

from pathlib import Path

W, H = 1600, 900
PINK, PINK_DEEP, PINK_TINT, PINK_LINE = "#D4537E", "#72243E", "#FBEAF0", "#F4C0D1"
CORAL, CORAL_TINT, CORAL_LINE, CORAL_DEEP = "#D85A30", "#FAECE7", "#F5C4B3", "#712B13"
INK, MUTED, PAGE, SURFACE, GRID = "#2C2C2A", "#6B6A65", "#FDF6F8", "#FFFFFF", "#D3D1C7"
SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', 'DejaVu Sans Mono', Menlo, Consolas, monospace"

out = []
add = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, dashed=False, fill=SURFACE, stroke=PINK_LINE):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"{d}/>')


def text(x, y, s, size=13, fill=INK, weight=400, font=SANS, anchor="start", style=""):
    st = f' font-style="{style}"' if style else ""
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


def chip(x, y, label, tone="pink", size=12, dashed=False):
    """Rounded monospace pill. Returns the x just past its right edge."""
    w = len(label) * size * 0.62 + 16
    fill, stroke, col = {
        "pink": (PINK_TINT, PINK_LINE, PINK_DEEP),
        "coral": (CORAL_TINT, CORAL_LINE, CORAL_DEEP),
        "plain": (SURFACE, GRID, MUTED),
    }[tone]
    d = ' stroke-dasharray="4 3"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{size + 9}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1"{d}/>')
    text(x + 8, y + size + 1.5, label, size=size, fill=col, font=MONO)
    return x + w + 7


def arrow(x1, y1, x2, y2, color=PINK, width=2):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}" marker-end="url(#arw)"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'role="img" aria-label="FLAMINGO pipeline">')
add('<title>FLAMINGO pipeline</title>')
add('<desc>Ten simulated biobank sites feed three estimator families — summary statistics, '
    'a pooled benchmark, and federated estimators — compared on a forest plot for the linear '
    'model and a dose-response curve for the non-linear model.</desc>')
add(f'<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto"><path d="M1 1L9 5L1 9" fill="none" stroke="{PINK}" '
    f'stroke-width="2"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="{PAGE}"/>')

# ---------------------------------------------------------------- title
text(40, 50, "FLAMINGO", size=34, fill=PINK_DEEP, weight=800)
text(232, 50, "Federated Non-Linear Mendelian Randomization", size=17, fill=MUTED)

COL_A, COL_B, COL_C = 40, 452, 1064
WA, WB, WC = 372, 572, 496
TOP = 80

# ============================================================ 1. causal model
box(COL_A, TOP, WA, 700)
text(COL_A + 20, TOP + 32, "1 · Shared causal model", size=17, fill=PINK_DEEP, weight=700)
text(COL_A + 20, TOP + 52, "identical structure at every site", size=12.5, fill=MUTED)

gy = TOP + 150
for bx, label in ((COL_A + 22, "SNPs G"), (COL_A + 142, "exposure X"), (COL_A + 262, "outcome Y")):
    add(f'<rect x="{bx}" y="{gy}" width="88" height="32" rx="6" fill="{PINK_TINT}" stroke="{PINK}" stroke-width="1"/>')
    text(bx + 44, gy + 21, label, size=12.5, fill=PINK_DEEP, font=MONO, anchor="middle")
ux = COL_A + 150
add(f'<rect x="{ux}" y="{gy - 72}" width="112" height="30" rx="6" fill="{CORAL_TINT}" stroke="{CORAL}" stroke-width="1"/>')
text(ux + 56, gy - 51, "confounder U", size=12.5, fill=CORAL_DEEP, font=MONO, anchor="middle")
arrow(COL_A + 112, gy + 16, COL_A + 138, gy + 16)
arrow(COL_A + 232, gy + 16, COL_A + 258, gy + 16)
arrow(ux + 28, gy - 40, COL_A + 190, gy - 6, CORAL, 1.6)
arrow(ux + 86, gy - 40, COL_A + 300, gy - 6, CORAL, 1.6)
text(COL_A + 125, gy - 6, "h²ₓ", size=12, fill=MUTED, font=MONO, anchor="middle")
text(COL_A + 150, gy - 22, "γₓ", size=12, fill=CORAL_DEEP, font=MONO, anchor="middle")
text(COL_A + 320, gy - 22, "γᵧ", size=12, fill=CORAL_DEEP, font=MONO, anchor="middle")
text(COL_A + 186, gy + 62, "f(X) = θ₁X + θ₂X²", size=14, fill=INK, font=MONO, anchor="middle")

y = gy + 100
text(COL_A + 20, y, "Outcome families", size=13, fill=INK, weight=600)
cx = COL_A + 20
for lab in ("linear", "quadratic"):
    cx = chip(cx, y + 10, lab)
cx = COL_A + 20
for lab in ("threshold", "u-shape"):
    cx = chip(cx, y + 38, lab)

y += 100
text(COL_A + 20, y, "In development", size=13, fill=CORAL_DEEP, weight=600)
text(COL_A + 133, y, "— no end-to-end results yet", size=12, fill=MUTED)
cx = COL_A + 20
for lab in ("cox (survival)", "binary logistic"):
    cx = chip(cx, y + 10, lab, tone="coral", dashed=True)

y += 80
text(COL_A + 20, y, "Ten biobank sites", size=13, fill=INK, weight=600)
# one tile per site; the coral bar is that site's share of the population, which
# is what differs between them
sx, sy = COL_A + 20, y + 14
for i, bar in enumerate((9, 12, 14, 17, 19, 15, 21, 23, 20, 22)):
    add(f'<rect x="{sx}" y="{sy}" width="29" height="30" rx="5" fill="{PINK_TINT}" '
        f'stroke="{PINK_LINE}" stroke-width="1"/>')
    text(sx + 14.5, sy + 13, f"{i + 1:02d}", size=10, fill=PINK_DEEP, font=MONO, anchor="middle")
    add(f'<rect x="{sx + 3}" y="{sy + 19}" width="{bar}" height="4" rx="2" fill="{CORAL}"/>')
    sx += 33
text(COL_A + 20, y + 68, "shared by every site:  θ₁, θ₂", size=12.5, fill=INK, font=MONO)
text(COL_A + 20, y + 88, "varies by site:  n 1k–10k,", size=12.5, fill=CORAL_DEEP, font=MONO)
text(COL_A + 20, y + 106, "h²ₓ, γₓ, γᵧ ~ Beta", size=12.5, fill=CORAL_DEEP, font=MONO)
text(COL_A + 20, y + 134, "writes  <shape>/siteNN.csv + manifest", size=12, fill=MUTED, font=MONO)
chip(COL_A + 20, y + 148, "simulate_federated_sites.py", tone="plain", size=11.5)

# ============================================================ 2/3/4 estimators
BH, GAP = 218, 23


def estimator_box(i, title, subtitle, rows, script, dashed=False, accent=PINK_DEEP):
    y0 = TOP + i * (BH + GAP)
    box(COL_B, y0, WB, BH, dashed=dashed)
    text(COL_B + 20, y0 + 31, title, size=17, fill=accent, weight=700)
    text(COL_B + 20, y0 + 51, subtitle, size=12.5, fill=MUTED)
    yy = y0 + 76
    for kind, a, b in rows:
        if kind == "kv":
            text(COL_B + 20, yy + 11, a, size=12.5, fill=MUTED, weight=600)
            text(COL_B + 132, yy + 11, b, size=12.5, fill=INK, font=MONO)
            yy += 24
        else:
            chip(COL_B + 20, yy, a, tone=b)
            yy += 28
    chip(COL_B + WB - 20 - len(script) * 11.5 * 0.62 - 16, y0 + BH - 28, script, tone="plain", size=11.5)
    return y0


estimator_box(
    0, "2 · Summary statistics", "per-SNP GWAS at each site — the conventional route",
    [("kv", "travels", "β, SE per SNP"),
     ("kv", "per site", "IVW over SNPs"),
     ("kv", "combined", "inverse-variance meta-analysis"),
     ("kv", "recovers", "one average slope")],
    "federated_summary_mr.py")
text(COL_B + 20, TOP + 195, "θ₂ is not identifiable from summary statistics alone",
     size=12, fill=CORAL_DEEP, style="italic")

y3 = estimator_box(
    1, "3 · Pooled — all sites, one model", "benchmark: needs every individual row in one place",
    [("kv", "linear", "concatenated 2SLS"),
     ("kv", "non-linear", "quadratic 2SLS on (X̂, X̂²)"),
     ("kv", "stage 1", "X̂ = SNP-predicted X, per site")],
    "federated_nonlinear_mr.py", dashed=True)
text(COL_B + 20, y3 + 171, "what a federation could not run — the yardstick, not a result",
     size=12, fill=MUTED, style="italic")

y4 = estimator_box(
    2, "4 · Federated — rows never move", "each site releases summed statistics or model updates",
    [("kv", "linear", "Fed-2SLS: exact 2SLS from sums"),
     ("kv", "non-linear", "Fed-2SRI: non-linear two-stage residual instruments")],
    "federated_exact_mr.py · job.py")
text(COL_B + 20, y4 + 171, "|FedMR − pooled| < 1e-10 — the same estimator, not an approximation",
     size=12, fill=PINK_DEEP, style="italic")

# ============================================================ 5. results
box(COL_C, TOP, WC, 700)
text(COL_C + 20, TOP + 32, "5 · Results", size=17, fill=PINK_DEEP, weight=700)
text(COL_C + 20, TOP + 52, "every estimator against the truth", size=12.5, fill=MUTED)

# ---- forest plot (linear)
fy = TOP + 80
text(COL_C + 20, fy, "linear model — forest plot", size=13.5, fill=INK, weight=600)
text(COL_C + 20, fy + 19, "sites and estimators on one axis", size=12, fill=MUTED)
fx0, fx1 = COL_C + 150, COL_C + WC - 40
truth = fx0 + (fx1 - fx0) * 0.56
ftop = fy + 34
add(f'<line x1="{truth}" y1="{ftop}" x2="{truth}" y2="{ftop + 212}" stroke="{INK}" '
    f'stroke-width="1.2" stroke-dasharray="5 4"/>')
text(truth, ftop - 6, "true θ", size=11.5, fill=INK, font=MONO, anchor="middle")
sites = [(-0.30, 0.20), (0.16, 0.15), (-0.12, 0.13), (0.24, 0.17), (-0.06, 0.11), (0.10, 0.14)]
ry = ftop + 16
for i, (off, half) in enumerate(sites):
    cx = truth + off * 70
    add(f'<line x1="{cx - half * 210:.1f}" y1="{ry}" x2="{cx + half * 210:.1f}" y2="{ry}" stroke="{PINK}" stroke-width="1.6"/>')
    add(f'<circle cx="{cx:.1f}" cy="{ry}" r="3.4" fill="{SURFACE}" stroke="{PINK}" stroke-width="1.6"/>')
    text(COL_C + 20, ry + 4, f"site{i + 1:02d}", size=11.5, fill=MUTED, font=MONO)
    ry += 20


def diamond(cx, cy, w, h, fill, stroke, dashed=False):
    d = ' stroke-dasharray="4 3"' if dashed else ""
    add(f'<path d="M{cx - w} {cy}L{cx} {cy - h}L{cx + w} {cy}L{cx} {cy + h}Z" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.6"{d}/>')


ry += 8
for label, off, fill, stroke, dash in (
        ("sumstats IVW", 0.10, PINK_TINT, PINK, False),
        ("pooled 2SLS", -0.04, SURFACE, INK, True),
        ("FedMR", -0.04, PINK, PINK_DEEP, False)):
    cx = truth + off * 70
    add(f'<line x1="{cx - 26}" y1="{ry}" x2="{cx + 26}" y2="{ry}" stroke="{stroke}" stroke-width="1.4"/>')
    diamond(cx, ry, 11, 6, fill, stroke, dash)
    text(COL_C + 20, ry + 4, label, size=11.5, fill=INK, font=MONO)
    ry += 24
text(COL_C + 150, ry + 6, "FedMR sits exactly on the pooled diamond", size=11.5, fill=PINK_DEEP, style="italic")

# ---- dose-response (non-linear)
dy = TOP + 380
add(f'<line x1="{COL_C + 20}" y1="{dy - 14}" x2="{COL_C + WC - 20}" y2="{dy - 14}" stroke="{PINK_LINE}" stroke-width="1"/>')
text(COL_C + 20, dy + 12, "non-linear model — dose-response curve", size=13.5, fill=INK, weight=600)
text(COL_C + 20, dy + 31, "the whole curve, not one average slope", size=12, fill=MUTED)
px0, py0, pw, ph = COL_C + 60, dy + 50, WC - 110, 190
add(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" stroke="{GRID}" stroke-width="1.2"/>')
add(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" stroke="{GRID}" stroke-width="1.2"/>')
text(px0 + pw / 2, py0 + ph + 30, "exposure X", size=11.5, fill=MUTED, anchor="middle")
add(f'<text x="{px0 - 16}" y="{py0 + ph / 2}" font-family="{SANS}" font-size="11.5" fill="{MUTED}" '
    f'text-anchor="middle" transform="rotate(-90 {px0 - 16} {py0 + ph / 2})">effect on Y</text>')


def curve(fn, color, width=2.4, dash=None, n=44):
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append(f"{px0 + t * pw:.1f},{py0 + ph - fn(t) * ph:.1f}")
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')


curve(lambda t: 0.08 + 0.80 * t * t, INK, 2.2, "6 4")          # truth
curve(lambda t: 0.10 + 0.74 * t * t, PINK, 2.6)                 # federated
curve(lambda t: 0.16 + 0.52 * t, GRID, 2.2)                     # sumstats average slope
ly = py0 + 14
for color, dash, label in ((INK, "6 4", "true curve"), (PINK, None, "federated / pooled"),
                           (GRID, None, "sumstats: average slope")):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{px0 + 16}" y1="{ly}" x2="{px0 + 44}" y2="{ly}" stroke="{color}" stroke-width="2.4"{d}/>')
    text(px0 + 52, ly + 4, label, size=11.5, fill=INK)
    ly += 18

# ============================================================ flow arrows
for i in range(3):
    ymid = TOP + i * (BH + GAP) + BH / 2
    arrow(COL_A + WA + 6, TOP + 350, COL_B - 6, ymid)
    arrow(COL_B + WB + 6, ymid, COL_C - 6, TOP + 350)

# ============================================================ legend
ly = TOP + 730
lx = COL_A + 4
for color, label, dashed in ((PINK, "pink — travels between sites", False),
                             (CORAL, "coral — varies per site, and the confounding", False),
                             (INK, "dashed — needs pooled individual rows (benchmark only)", True)):
    d = ' stroke-dasharray="5 4"' if dashed else ""
    add(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{color}" stroke-width="2.6"{d}/>')
    text(lx + 34, ly + 4, label, size=12.5, fill=MUTED)
    lx += len(label) * 6.6 + 78
add('</svg>')

dest = Path(__file__).with_name("flamingo_pipeline.svg")
dest.write_text("\n".join(out))

# Chrome applies a default body margin to a bare .svg, which clips it; wrapping
# the markup strips that.
wrapper = Path(__file__).with_name("_render.html")
wrapper.write_text(
    "<!doctype html><meta charset=utf-8>"
    "<style>html,body{margin:0;padding:0}svg{display:block}</style>\n" + "\n".join(out))

png = dest.with_suffix(".png")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCALE = 2  # 2x so the PNG stays sharp projected or zoomed

if Path(CHROME).exists():
    import subprocess
    # The window is given extra height on purpose: at exactly H the viewport
    # comes back a little short and Chrome clips the last ~90px. Render tall,
    # then crop to the canvas.
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={SCALE}",
                    f"--window-size={W},{H + 200}",
                    f"--screenshot={png}", str(wrapper)],
                   check=True, capture_output=True, cwd=dest.parent)
    from PIL import Image
    Image.open(png).crop((0, 0, W * SCALE, H * SCALE)).save(png, optimize=True)
    wrapper.unlink()
    print(f"wrote {dest.name} and {png.name} ({W * SCALE}x{H * SCALE}, "
          f"{png.stat().st_size / 1024:.0f} KB)")
else:
    print(f"wrote {dest.name}; Chrome not found, PNG not rendered")
