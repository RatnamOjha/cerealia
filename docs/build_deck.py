"""
Generate the KrishiMitra project review deck.

Kept as a script rather than a hand-built file so the numbers stay tied to what
the model actually produces: re-run after retraining and the metrics slide
updates itself.

Run:  ./backend/.venv/bin/python docs/build_deck.py
Out:  docs/KrishiMitra_Project_Review.pptx
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
OUT = ROOT / "docs" / "KrishiMitra_Project_Review.pptx"

METRICS = json.loads((ROOT / "backend" / "models" / "metrics.json").read_text())

BG = RGBColor(0x0B, 0x14, 0x18)
PANEL = RGBColor(0x15, 0x28, 0x2E)
GREEN = RGBColor(0x6E, 0xCF, 0x94)
GREEN_D = RGBColor(0x4C, 0x9F, 0x70)
TEXT = RGBColor(0xE8, 0xF1, 0xEE)
DIM = RGBColor(0x8F, 0xA8, 0xA5)
AMBER = RGBColor(0xE0, 0xA4, 0x58)
RED = RGBColor(0xD1, 0x68, 0x5C)

W, H = Inches(13.333), Inches(7.5)
FONT = "Helvetica Neue"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def slide(prs: Presentation):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def textbox(s, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size, color=TEXT, bold=False, space_before=0, space_after=6,
         align=PP_ALIGN.LEFT, first=False, italic=False, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = FONT
    return p


def title(s, text, sub=None):
    tf = textbox(s, Inches(0.75), Inches(0.45), Inches(11.8), Inches(1.1))
    para(tf, text, 34, TEXT, bold=True, first=True, space_after=2)
    if sub:
        para(tf, sub, 15, DIM, space_after=0)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.62), Inches(1.5), Pt(3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GREEN
    bar.line.fill.background()
    bar.shadow.inherit = False


def card(s, x, y, w, h, fill=PANEL, border=None):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    c.adjustments[0] = 0.05
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    if border:
        c.line.color.rgb = border
        c.line.width = Pt(1.25)
    else:
        c.line.fill.background()
    c.shadow.inherit = False
    return c


def stat_card(s, x, y, w, h, value, label, color=GREEN):
    card(s, x, y, w, h)
    tf = textbox(s, x + Inches(0.15), y + Inches(0.18), w - Inches(0.3), h - Inches(0.3),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, value, 30, color, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=2)
    para(tf, label, 11, DIM, align=PP_ALIGN.CENTER, space_after=0)


def bullets(s, x, y, w, h, items, size=15, gap=11):
    tf = textbox(s, x, y, w, h)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            head, body = item
            p = para(tf, head, size + 1, GREEN, bold=True, first=(i == 0),
                     space_before=0 if i == 0 else gap, space_after=3)
            para(tf, body, size - 1.5, DIM, space_after=0, line=1.25)
        else:
            para(tf, "•  " + item, size, TEXT, first=(i == 0),
                 space_before=0 if i == 0 else gap, space_after=0, line=1.25)
    return tf


def footer(s, n):
    tf = textbox(s, Inches(11.9), Inches(6.92), Inches(0.9), Inches(0.4), align=PP_ALIGN.RIGHT)
    para(tf, str(n), 10, RGBColor(0x4A, 0x60, 0x5E), first=True, align=PP_ALIGN.RIGHT)


def picture(s, name, x, y, w):
    path = SHOTS / name
    if path.exists():
        s.shapes.add_picture(str(path), x, y, width=w)


# --------------------------------------------------------------------------


def build() -> None:
    prs = new_deck()
    n = 0

    # 1 -- Title
    s = slide(prs)
    tf = textbox(s, Inches(1.0), Inches(2.3), Inches(11.3), Inches(2.6))
    para(tf, "KrishiMitra", 62, TEXT, bold=True, first=True, space_after=4)
    para(tf, "AI-Based Crop Recommendation for Indian Farmers", 25, GREEN, space_after=16)
    para(tf, "Which crop should I plant — and what will it actually earn me?",
         16, DIM, italic=True, space_after=0)
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(5.15), Inches(2.2), Pt(3))
    line.fill.solid(); line.fill.fore_color.rgb = GREEN; line.line.fill.background()
    line.shadow.inherit = False
    tf = textbox(s, Inches(1.0), Inches(5.5), Inches(11), Inches(1))
    para(tf, "Minor Project  ·  Progress Review", 14, DIM, first=True, space_after=3)
    para(tf, "Interactive map of India  ·  36 states & UTs  ·  22 crops  ·  grounded scheme advisory",
         12.5, RGBColor(0x5F, 0x7D, 0x7A), space_after=0)

    # 2 -- Problem
    n += 1; s = slide(prs); footer(s, n)
    title(s, "The Problem", "Crop choice is the single highest-leverage decision a farmer makes each season — and it is usually made blind")
    items = [
        ("Decisions are made on habit, not data",
         "Farmers largely repeat last year's crop or copy neighbours. Soil Health Cards exist but arrive as a 12-parameter lab report with no clear action attached."),
        ("The advice that exists answers the wrong question",
         "Existing tools predict what CAN grow. A farmer already knows rice grows in his village. He needs to know what is worth growing this year."),
        ("Nobody prices the decision",
         "Suitability says nothing about cost of cultivation, mandi price, irrigation burden or how long until the first harvest."),
        ("Support schemes go unclaimed",
         "Central schemes cover insurance, credit, irrigation and solar pumps — but eligibility rules are buried in PDFs across a dozen portals."),
    ]
    bullets(s, Inches(0.75), Inches(2.15), Inches(11.8), Inches(4.4), items, size=15, gap=15)

    # 3 -- Objective
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Objective", "Turn a crop classifier into an advisor a farmer could actually act on")
    card(s, Inches(0.75), Inches(2.1), Inches(11.8), Inches(1.25), fill=RGBColor(0x14, 0x2E, 0x28),
         border=GREEN_D)
    tf = textbox(s, Inches(1.1), Inches(2.28), Inches(11.1), Inches(0.95), anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "Given where a farmer is and what their soil looks like, rank the crops they could plant by "
             "what each is realistically worth — and explain why.", 17, TEXT, bold=True, first=True,
         space_after=0, line=1.2)
    cols = [
        ("Pick a region", "3D globe → map of India.\nClick any state to load its\nsoil and climate profile."),
        ("Rank the crops", "Not just suitability —\nprofit per hectare, water\nneed and risk, combined."),
        ("Explain the answer", "Show which conditions match,\nwhich don't, and the fertiliser\nneeded to close the gap."),
        ("Answer the follow-up", "A grounded chatbot for the\ngovernment schemes that\napply to that decision."),
    ]
    x = Inches(0.75)
    for head, body in cols:
        card(s, x, Inches(3.65), Inches(2.78), Inches(2.5))
        tf = textbox(s, x + Inches(0.25), Inches(3.9), Inches(2.3), Inches(2.1))
        para(tf, head, 15, GREEN, bold=True, first=True, space_after=8)
        para(tf, body, 12, DIM, space_after=0, line=1.3)
        x += Inches(3.0)

    # 4 -- Why not just a classifier
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Why a Classifier Alone Is Not Enough",
          "The standard project stops at 99% accuracy. That number is misleading and the output is not advice.")
    card(s, Inches(0.75), Inches(2.1), Inches(5.7), Inches(4.3), border=RED)
    tf = textbox(s, Inches(1.05), Inches(2.35), Inches(5.1), Inches(3.9))
    para(tf, "The usual approach", 16, RED, bold=True, first=True, space_after=10)
    for t in ["7 features in → 1 crop name out",
              "Trained on a synthetic dataset with clean,\nwell-separated class boundaries",
              "99% accuracy is easy and means little",
              "No geography: the model will suggest\nmango for Ladakh",
              "No economics: never asks what the crop\ncosts to grow or sells for",
              "No explanation the farmer can check"]:
        para(tf, "×   " + t, 13, DIM, space_after=9, line=1.25)

    card(s, Inches(6.85), Inches(2.1), Inches(5.7), Inches(4.3), border=GREEN_D)
    tf = textbox(s, Inches(7.15), Inches(2.35), Inches(5.1), Inches(3.9))
    para(tf, "This project", 16, GREEN, bold=True, first=True, space_after=10)
    for t in ["The classifier is one signal out of four",
              "A regional cultivation prior supplies the\ngeography the dataset lacks",
              "Ranked on risk-adjusted expected return,\nnot headline profit",
              "Water demand checked against actual\nrainfall — irrigation gap in mm",
              "Every recommendation shows the ideal\nband for each condition",
              "Fertiliser gap costed in bags and rupees"]:
        para(tf, "✓   " + t, 13, TEXT, space_after=9, line=1.25)

    # 5 -- Architecture
    n += 1; s = slide(prs); footer(s, n)
    title(s, "System Architecture")
    layers = [
        ("FRONTEND", "React + Vite", "d3-geo canvas globe · India state map (85 KB, bundled offline) · recommendation panel · chat drawer", GREEN),
        ("API", "FastAPI", "/api/states · /api/overview · /api/recommend/state · /api/recommend/custom · /api/schemes · /api/chat", RGBColor(0x5B, 0x8A, 0xB0)),
        ("ENGINE", "recommender.py + chatbot.py", "4-signal ranking · percentile-envelope explainability · fertiliser planner · grounded scheme retrieval", AMBER),
        ("DATA", "Curated + trained artefacts", "2,200-row training set · 36 state agro-climatic profiles · crop economics · cultivation prior · 12 schemes", RGBColor(0x8E, 0x6C, 0xAE)),
    ]
    y = Inches(2.15)
    for tag, name, detail, color in layers:
        card(s, Inches(0.75), y, Inches(11.8), Inches(1.02))
        chip = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), y + Inches(0.28), Inches(1.35), Inches(0.42))
        chip.adjustments[0] = 0.3
        chip.fill.solid(); chip.fill.fore_color.rgb = color; chip.line.fill.background()
        chip.shadow.inherit = False
        ctf = chip.text_frame; ctf.word_wrap = False
        ctf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = ctf.paragraphs[0].add_run(); r.text = tag
        r.font.size = Pt(10); r.font.bold = True; r.font.color.rgb = BG; r.font.name = FONT
        tf = textbox(s, Inches(2.6), y + Inches(0.16), Inches(9.7), Inches(0.8))
        para(tf, name, 15, TEXT, bold=True, first=True, space_after=2)
        para(tf, detail, 11.5, DIM, space_after=0)
        y += Inches(1.12)

    # 6 -- The four signals
    n += 1; s = slide(prs); footer(s, n)
    title(s, "The Ranking Engine", "Four signals, weighted explicitly — so the policy is auditable rather than buried in a model")
    sigs = [
        ("1. Agro-climatic fitness", "45%", "Can it survive here?",
         "RandomForest probability blended with per-crop 10th–90th percentile envelopes. The forest is accurate but spiky; the envelope gives a smooth signal across all 22 crops.", GREEN),
        ("2. Economics", "30%", "What does it earn?",
         "Yield × price − cost, annualised. Short crops scaled by cycles per year; perennials carry amortised establishment cost, so a 70-day moong is comparable to a 30-year mango orchard.", AMBER),
        ("3. Water feasibility", "15%", "Can they irrigate it?",
         "Crop water requirement vs 70% of local rainfall. Returns the irrigation gap in mm and a rainfed / light / moderate / heavy verdict.", RGBColor(0x5B, 0x8A, 0xB0)),
        ("4. Risk", "10%", "How exposed are they?",
         "Price volatility, perishability, capital lock-in and years to first harvest.", RED),
    ]
    y = Inches(2.1)
    for name, weight, q, detail, color in sigs:
        card(s, Inches(0.75), y, Inches(11.8), Inches(1.0))
        tf = textbox(s, Inches(1.05), y + Inches(0.12), Inches(3.3), Inches(0.8))
        para(tf, name, 14.5, color, bold=True, first=True, space_after=1)
        para(tf, q, 11, DIM, italic=True, space_after=0)
        tf = textbox(s, Inches(4.5), y + Inches(0.14), Inches(6.6), Inches(0.8))
        para(tf, detail, 11.5, TEXT, first=True, space_after=0, line=1.2)
        tf = textbox(s, Inches(11.25), y + Inches(0.24), Inches(1.1), Inches(0.6), align=PP_ALIGN.RIGHT)
        para(tf, weight, 22, color, bold=True, first=True, align=PP_ALIGN.RIGHT, space_after=0)
        y += Inches(1.1)
    tf = textbox(s, Inches(0.75), Inches(6.62), Inches(11.8), Inches(0.6))
    para(tf, "Fitness also acts as a gate: a crop the region does not cultivate is removed outright, "
             "however profitable it looks.", 12, DIM, italic=True, first=True, space_after=0)

    # 7 -- Model results
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Model & Results", f"RandomForest · 300 trees · {METRICS['dataset']['n_rows']:,} samples · {METRICS['dataset']['n_classes']} crops")
    w, gap = Inches(2.78), Inches(3.0)
    stat_card(s, Inches(0.75), Inches(2.1), w, Inches(1.5),
              f"{METRICS['holdout_accuracy']*100:.2f}%", "Hold-out accuracy")
    stat_card(s, Inches(0.75) + gap, Inches(2.1), w, Inches(1.5),
              f"{METRICS['cv_accuracy_mean']*100:.2f}%", f"5-fold CV (± {METRICS['cv_accuracy_std']*100:.2f}%)")
    stat_card(s, Inches(0.75) + 2*gap, Inches(2.1), w, Inches(1.5),
              f"{METRICS['misclassified_test_samples']} / {METRICS['test_set_size']}", "Misclassified", AMBER)
    stat_card(s, Inches(0.75) + 3*gap, Inches(2.1), w, Inches(1.5),
              f"{METRICS['train_seconds']:.2f}s", "Training time")

    card(s, Inches(0.75), Inches(3.85), Inches(5.7), Inches(2.55))
    tf = textbox(s, Inches(1.05), Inches(4.05), Inches(5.1), Inches(2.2))
    para(tf, "FEATURE IMPORTANCE", 11, DIM, bold=True, first=True, space_after=10)
    for feat, val in list(METRICS["feature_importance"].items()):
        p = tf.add_paragraph(); p.space_after = Pt(5)
        r = p.add_run(); r.text = f"{feat:<13}"
        r.font.size = Pt(12); r.font.color.rgb = TEXT; r.font.name = "Menlo"
        r2 = p.add_run(); r2.text = "█" * max(1, int(val * 52)) + f"  {val:.3f}"
        r2.font.size = Pt(10); r2.font.color.rgb = GREEN; r2.font.name = "Menlo"

    card(s, Inches(6.85), Inches(3.85), Inches(5.7), Inches(2.55), border=AMBER)
    tf = textbox(s, Inches(7.15), Inches(4.05), Inches(5.1), Inches(2.2))
    para(tf, "WHY NO SMOTE", 11, AMBER, bold=True, first=True, space_after=10)
    para(tf, f"The dataset is exactly balanced — {METRICS['dataset']['min_class_count']} samples "
             f"for every one of the {METRICS['dataset']['n_classes']} crops.", 13, TEXT, space_after=8, line=1.25)
    para(tf, "Resampling a balanced dataset injects synthetic noise without correcting any imbalance. "
             "train.py asserts the balance rather than assuming it.", 12, DIM, space_after=8, line=1.25)
    para(tf, "Knowing when not to apply a technique is part of the result.", 12, GREEN, italic=True, space_after=0)

    # 8 -- Demo: globe + map
    n += 1; s = slide(prs); footer(s, n)
    title(s, "The Interface", "Globe → map of India → click a state")
    picture(s, "01-globe.png", Inches(0.75), Inches(2.2), Inches(5.75))
    picture(s, "02-map-overview.png", Inches(6.8), Inches(2.2), Inches(5.75))
    tf = textbox(s, Inches(0.75), Inches(5.95), Inches(5.75), Inches(1.2))
    para(tf, "Rendered with d3-geo on canvas", 12.5, GREEN, bold=True, first=True, space_after=3)
    para(tf, "No three.js, no CDN textures — the globe works with no network. "
             "Clicking animates rotation to India's centroid.", 11, DIM, space_after=0, line=1.2)
    tf = textbox(s, Inches(6.8), Inches(5.95), Inches(5.75), Inches(1.2))
    para(tf, "36 states, coloured by top crop type", 12.5, GREEN, bold=True, first=True, space_after=3)
    para(tf, "Dissolved from a 760-district GeoJSON and simplified 4 MB → 85 KB, "
             "bundled locally so the map never fetches at runtime.", 11, DIM, space_after=0, line=1.2)

    # 9 -- Demo: recommendation
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Recommendation Output", "Maharashtra — click a state and the panel loads its ranked crops")
    picture(s, "03-maharashtra.png", Inches(0.75), Inches(2.1), Inches(7.6))
    tf = textbox(s, Inches(8.6), Inches(2.2), Inches(3.95), Inches(4.4))
    for head, body in [
        ("Ranked crop list", "Each card shows climate fit, water need and net profit per year, scaled to the farm size slider."),
        ("MSP badge", "Flags crops with a guaranteed government floor price — lower price risk."),
        ("Honest confidence", "Every card is labelled high / moderate / low so a weak match is never presented as certainty."),
        ("Gate transparency", "The panel states how many crops were excluded as not cultivated in that region."),
    ]:
        para(tf, head, 13.5, GREEN, bold=True, first=(head == "Ranked crop list"),
             space_before=0 if head == "Ranked crop list" else 14, space_after=4)
        para(tf, body, 11.5, DIM, space_after=0, line=1.25)

    # 10 -- Explainability
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Explainability", "A farmer will not act on a number they cannot check")
    picture(s, "04-explainability.png", Inches(0.75), Inches(2.1), Inches(7.6))
    tf = textbox(s, Inches(8.6), Inches(2.2), Inches(3.95), Inches(4.4))
    for head, body in [
        ("Condition vs ideal band", "Every reading is compared against that crop's observed 10th–90th percentile range, with mismatches sorted to the top."),
        ("Full cost breakdown", "Gross revenue, operating cost, amortised setup, and the risk-adjusted figure — not just a single number."),
        ("Costed fertiliser plan", "The NPK gap converted into kilograms of urea / DAP / MOP, 50 kg bags, and rupees."),
        ("Verifiable", "An extension officer can check every line against the Soil Health Card."),
    ]:
        para(tf, head, 13.5, GREEN, bold=True, first=(head == "Condition vs ideal band"),
             space_before=0 if head == "Condition vs ideal band" else 14, space_after=4)
        para(tf, body, 11.5, DIM, space_after=0, line=1.25)

    # 11 -- Chatbot
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Scheme Advisory Chatbot", "Grounded on 12 central schemes — Grok (x.ai) with a local retrieval fallback")
    picture(s, "05-chatbot.png", Inches(0.75), Inches(2.1), Inches(7.6))
    tf = textbox(s, Inches(8.6), Inches(2.2), Inches(3.95), Inches(4.4))
    for head, body in [
        ("Retrieval before generation", "The question first retrieves matching schemes; only that verified text is sent to the model as context."),
        ("Constrained on purpose", "A hallucinated subsidy percentage costs a farmer real money, so the model summarises verified text instead of recalling from training."),
        ("Degrades, never fails", "With no API key the same retrieval renders a structured answer directly — the demo runs with no network."),
        ("Context-aware", "The selected state and top crop are passed in, so the farmer never restates them."),
    ]:
        para(tf, head, 13.5, GREEN, bold=True, first=(head == "Retrieval before generation"),
             space_before=0 if head == "Retrieval before generation" else 13, space_after=4)
        para(tf, body, 11.5, DIM, space_after=0, line=1.25)

    # 12 -- Validation
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Does It Match Reality?", "Recommendations reproduce real Indian cropping patterns — none of them hard-coded as answers")
    rows = [
        ("Assam", "Jute, Rice", "India's jute and paddy belt", GREEN),
        ("West Bengal", "Jute, Rice, Banana", "Largest jute-producing state", GREEN),
        ("Kerala", "Banana, Rice, Coconut, Coffee", "Matches the actual cropping pattern", GREEN),
        ("Maharashtra", "Mango, Tur, Orange", "Konkan mango, Vidarbha tur, Nagpur orange", GREEN),
        ("Rajasthan", "Moth beans, Watermelon, Chickpea", "Arid-zone crops", GREEN),
        ("Himachal Pradesh", "Maize, Orange, Rajma", "Maize is HP's largest crop by area", GREEN),
    ]
    y = Inches(2.2)
    hdr = textbox(s, Inches(1.0), y, Inches(11.4), Inches(0.3))
    p = hdr.paragraphs[0]
    for txt, wid in [("STATE", 2.6), ("TOP RECOMMENDATIONS", 4.6), ("REALITY CHECK", 4.2)]:
        r = p.add_run(); r.text = txt.ljust(int(wid * 5.2))
        r.font.size = Pt(9.5); r.font.bold = True; r.font.color.rgb = DIM; r.font.name = "Menlo"
    y += Inches(0.42)
    for state, crops, note, color in rows:
        card(s, Inches(0.75), y, Inches(11.8), Inches(0.6))
        tf = textbox(s, Inches(1.0), y + Inches(0.13), Inches(2.5), Inches(0.4))
        para(tf, state, 13, TEXT, bold=True, first=True, space_after=0)
        tf = textbox(s, Inches(3.6), y + Inches(0.14), Inches(4.4), Inches(0.4))
        para(tf, crops, 12.5, color, first=True, space_after=0)
        tf = textbox(s, Inches(8.1), y + Inches(0.15), Inches(4.3), Inches(0.4))
        para(tf, note, 11.5, DIM, first=True, space_after=0)
        y += Inches(0.68)

    # 13 -- The Punjab finding
    n += 1; s = slide(prs); footer(s, n)
    title(s, "It Also Surfaces Real Problems", "The engine does not hide agronomic tension — it reports it")
    card(s, Inches(0.75), Inches(2.15), Inches(11.8), Inches(2.1), fill=RGBColor(0x2A, 0x1C, 0x18), border=AMBER)
    tf = textbox(s, Inches(1.2), Inches(2.45), Inches(11), Inches(1.6))
    para(tf, "Punjab paddy scores only 0.39 agro-climatic fitness", 22, AMBER, bold=True, first=True, space_after=8)
    para(tf, "649 mm of annual rainfall cannot support rice. Punjab grows it anyway, through groundwater "
             "extraction — and the engine flags a 346 mm irrigation gap with a \"check groundwater status\" "
             "verdict. The state's documented water crisis appears directly in the model output.",
         14, TEXT, space_after=0, line=1.3)
    two = [
        ("What a naive system would do",
         "Rank on headline profit and recommend pomegranate for all 36 states — including Punjab, where it "
         "scores 0% suitability. This is not hypothetical: it is what our first implementation did, and it is "
         "reproducible by setting the profit weight to 0.9."),
        ("What fixed it",
         "Rank on risk-adjusted expected return — profit weighted by agro-climatic fitness — and gate out crops "
         "the region does not cultivate. A crop paying ₹6 lakh/ha that half-fits the climate is not worth more "
         "than one paying ₹3 lakh that thrives."),
    ]
    x = Inches(0.75)
    for head, body in two:
        card(s, x, Inches(4.5), Inches(5.75), Inches(2.0))
        tf = textbox(s, x + Inches(0.28), Inches(4.72), Inches(5.2), Inches(1.7))
        para(tf, head, 13.5, GREEN, bold=True, first=True, space_after=6)
        para(tf, body, 11.5, DIM, space_after=0, line=1.28)
        x += Inches(6.05)

    # 14 -- Current status
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Current Progress", "Working end to end — backend, engine, interface and chatbot")
    done = [
        "RandomForest trained, evaluated and versioned (99.50% CV)",
        "Four-signal ranking engine with regional cultivation prior",
        "Per-crop economics for all 22 crops (MSP, yield, cost, water, risk)",
        "Agro-climatic profiles for all 36 states and UTs",
        "Explainability layer + costed fertiliser planner",
        "FastAPI service — 7 endpoints, CORS, validation",
        "React frontend — globe, map, panel, chat drawer",
        "Grounded scheme chatbot with offline fallback",
        "Verified end to end in headless Chrome, zero console errors",
        "Pushed to GitHub with full documentation",
    ]
    pending = [
        "Grok API key wiring (fallback works today)",
        "District-level resolution",
        "Live weather and mandi price feeds",
        "ESP32 soil sensor hardware",
        "Hindi / regional language support",
    ]
    card(s, Inches(0.75), Inches(2.1), Inches(7.3), Inches(4.35), border=GREEN_D)
    tf = textbox(s, Inches(1.05), Inches(2.3), Inches(6.8), Inches(4.0))
    para(tf, "DONE", 12, GREEN, bold=True, first=True, space_after=10)
    for d in done:
        para(tf, "✓   " + d, 12.5, TEXT, space_after=8, line=1.15)
    card(s, Inches(8.45), Inches(2.1), Inches(4.1), Inches(4.35))
    tf = textbox(s, Inches(8.75), Inches(2.3), Inches(3.6), Inches(4.0))
    para(tf, "NEXT", 12, AMBER, bold=True, first=True, space_after=10)
    for d in pending:
        para(tf, "○   " + d, 12, DIM, space_after=10, line=1.2)

    # 15 -- Requirements
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Requirements")
    groups = [
        ("Software", ["Python 3.12 · FastAPI · Uvicorn", "scikit-learn · pandas · NumPy · joblib",
                      "React 18 · Vite · d3-geo · topojson", "httpx · python-pptx · shapely"], GREEN),
        ("Data", ["Crop_recommendation.csv (2,200 × 22)", "IMD rainfall & temperature normals",
                  "Soil Health Card NPK/pH aggregates", "CACP MSP 2024-25 · Agmarknet prices",
                  "Agricultural Statistics at a Glance"], RGBColor(0x5B, 0x8A, 0xB0)),
        ("External APIs", ["Grok / x.ai — scheme chatbot (free tier)", "Open-Meteo — live weather (planned)",
                           "Agmarknet / data.gov.in — mandi prices (planned)", "Bhashini — Indian languages (planned)"], AMBER),
        ("Hardware", ["Laptop — runs entirely offline for the demo", "ESP32 + RS485 7-in-1 NPK soil sensor (planned)",
                      "DHT22 temperature / humidity module", "Feeds live readings in place of state averages"], RGBColor(0x8E, 0x6C, 0xAE)),
    ]
    x, y = Inches(0.75), Inches(2.15)
    for i, (head, items, color) in enumerate(groups):
        cx = x + Inches(6.05) * (i % 2)
        cy = y + Inches(2.25) * (i // 2)
        card(s, cx, cy, Inches(5.75), Inches(2.05))
        tf = textbox(s, cx + Inches(0.28), cy + Inches(0.2), Inches(5.2), Inches(1.75))
        para(tf, head.upper(), 11.5, color, bold=True, first=True, space_after=8)
        for it in items:
            para(tf, "·  " + it, 11.5, DIM, space_after=5, line=1.15)

    # 16 -- Roadmap
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Future Plans", "Three horizons")
    horizons = [
        ("NEAR TERM", GREEN, [
            "District-level resolution (Nashik ≠ Vidarbha)",
            "Live weather via Open-Meteo",
            "Live mandi prices via Agmarknet",
            "SHAP explanations alongside the bands",
            "Hindi + regional languages via Bhashini",
            "Voice interface for low-literacy users",
        ]),
        ("MEDIUM TERM", AMBER, [
            "Crop rotation planning across seasons",
            "Groundwater sustainability scoring (CGWB)",
            "Mandi price forecasting",
            "Leaf-disease detection from photos",
            "ESP32 + NPK sensor for live soil readings",
            "WhatsApp bot for non-smartphone farmers",
        ]),
        ("LONGER TERM", RGBColor(0x8E, 0x6C, 0xAE), [
            "Sentinel-2 NDVI crop health monitoring",
            "Land allocation optimisation under budget",
            "FPO and district-officer dashboards",
            "Glut / shortage early warning from intent",
            "Carbon credit and sustainability scoring",
            "Credit scoring for KCC pre-qualification",
        ]),
    ]
    x = Inches(0.75)
    for head, color, items in horizons:
        card(s, x, Inches(2.15), Inches(3.83), Inches(4.35))
        tf = textbox(s, x + Inches(0.28), Inches(2.38), Inches(3.3), Inches(4.0))
        para(tf, head, 12, color, bold=True, first=True, space_after=12)
        for it in items:
            para(tf, "·  " + it, 12, TEXT if color == GREEN else DIM, space_after=11, line=1.2)
        x += Inches(4.05)

    # 17 -- Limitations
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Known Limitations", "Stated plainly, because they shape the roadmap")
    lims = [
        ("The training dataset is synthetic",
         "Clean, well-separated feature envelopes are why 99.5% accuracy is easy. That figure should not be read as real-world performance — and the dataset carries no geography at all, which is exactly why the cultivation prior exists."),
        ("Apple is under-ranked for Himachal",
         "The dataset places apple's temperature band at 21–24 °C, which does not reflect real apple agronomy — apple needs roughly 1,000 winter chill hours. A vernalisation feature is needed."),
        ("State-level granularity",
         "Nashik and Vidarbha are both simply \"Maharashtra\" today."),
        ("Annual averages, not seasonal",
         "Crops grow in specific seasons; annual means blur kharif and rabi together."),
        ("Wheat and sugarcane are absent",
         "Two of India's largest crops are simply not among the 22 in the dataset."),
    ]
    bullets(s, Inches(0.75), Inches(2.15), Inches(11.8), Inches(4.4), lims, size=14, gap=13)

    # 18 -- Close
    n += 1; s = slide(prs); footer(s, n)
    title(s, "Summary")
    card(s, Inches(0.75), Inches(2.1), Inches(11.8), Inches(1.5), fill=RGBColor(0x14, 0x2E, 0x28), border=GREEN_D)
    tf = textbox(s, Inches(1.2), Inches(2.35), Inches(11), Inches(1.1), anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "A crop recommender that answers the question a farmer actually has —", 19, TEXT, bold=True,
         first=True, space_after=4)
    para(tf, "not \"what can grow here?\" but \"what is worth growing, and why?\"", 19, GREEN, bold=True, space_after=0)
    stats = [("36", "states & UTs"), ("22", "crops ranked"), ("99.5%", "CV accuracy"),
             ("12", "schemes indexed"), ("4", "ranking signals")]
    x = Inches(0.75)
    for val, lbl in stats:
        stat_card(s, x, Inches(3.9), Inches(2.2), Inches(1.35), val, lbl)
        x += Inches(2.4)
    tf = textbox(s, Inches(0.75), Inches(5.6), Inches(11.8), Inches(1.2))
    para(tf, "github.com/RatnamOjha/krishi-mitra", 15, GREEN, bold=True, first=True, space_after=5)
    para(tf, "Backend, engine, frontend and chatbot working end to end · runs fully offline for the demo",
         12.5, DIM, space_after=0)

    prs.save(OUT)
    print(f"Saved {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
