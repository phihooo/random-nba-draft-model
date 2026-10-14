# Can college stats predict NBA stardom? — a plain-English explainer

*A beginner-friendly walkthrough of the analysis in `new.py` and the composite-index
plots. No statistics background assumed.*

---

## The question

When a player is drafted out of college, can his **college numbers** tell us how good
he'll become in the NBA?

Short answer: **they set a floor (who probably won't make it) but can't pick the
ceiling (who will be great).** The rest of this document explains how we know that.

---

## The one idea you need: correlation (r)

Correlation, written **r**, measures: *when one thing goes up, does the other tend to
go up too?* It runs from **−1 to +1**.

| r | Meaning |
|---|---|
| +1 | Perfect — X up always means Y up (e.g. inches vs centimetres) |
| 0 | No relationship — X tells you nothing about Y |
| −1 | Perfect opposite — X up always means Y down |
| ~0.3 | Weak/moderate — a real tendency, lots of exceptions |
| ~0.5 | Moderate — the bar this project treats as "worth taking seriously" |

**Trick:** square r to get the share of the outcome it "explains."
- r = 0.3 → 0.09 → about **9%** explained (91% is other stuff)
- r = 0.43 → about **18%** explained
- r = 0.5 → **25%** explained

Even a "moderate" correlation leaves most of the story untold.

---

## What we predict: "peak 3-year VORP"

**VORP** is a standard basketball stat for how much value a player added over a
replacement-level player. Higher = better.

We use **peak 3-year VORP** = a player's best three straight seasons, averaged. Why the
peak and not the whole career? Because careers get cut short by injuries and bad luck
that **no college stat could predict**. Looking at the peak asks the fairer question:
"how good was he *at his best*?"

---

## Finding 1 — no single college stat is a good predictor

Correlating each college stat, one at a time, with peak VORP:

- Best single clue: **draft pick**, r ≈ −0.39
  (negative only because pick #1 is *better* than pick #50 — smaller = better)
- Best *college* stat: **BPM** (an all-in-one quality score), r ≈ 0.31
- Most stats: near **0** (raw height ≈ 0.00)

**No single stat reached the r = 0.5 bar.** There is no magic stat.

---

## Finding 2 — many stats secretly measure the same thing

Points, usage, free throws made... all really capture *"this guy scores a lot."*
They're one clue wearing several hats. Throwing all 40 into a model double-counts.

**The checkerboard graph** (`plots/college_stat_correlation_clusters.png`) maps this:

- **Red square** = the two stats rise and fall together (duplicates)
- **Blue square** = they move in opposite directions
- **Pale** = unrelated

Rows/columns are reordered so similar stats sit together — hence the **red blocks along
the diagonal**. Each red block is a "family" of near-duplicate stats (a scoring family,
a rebounding/size family, a shooting-accuracy family, etc.). Its job: show which stats
are redundant.

---

## Finding 3 — combine the families, and the signal roughly doubles

Squash each family into one combined score (a "composite," basically an average), then
model those:

- Best single stat: r ≈ 0.31
- Combined sensibly (college only): r ≈ **0.43**
- Adding draft position: r ≈ 0.47

Combining redundant clues **roughly doubles** predictive power — weak → moderate. But it
**still doesn't cross 0.5** on college stats alone. That ceiling is real.

---

## The scatter plots — what "combining" looks like

`plots/composite_index_vs_peak_vorp.png` and
`plots/composite_equal_vs_weighted.png`.

Each dot = one player.
- **Left–right** = his college index (right = stronger college profile)
- **Up–down** = how good he actually became (up = better)
- **Red dots** = the 20 best NBA players in the data
- **Red line** = the overall trend (slopes up = better profile tends to mean better pro)

Two things to see:
1. **Every red superstar is on the right** — no star came from a weak profile. A strong
   profile is **necessary**.
2. **The right side is also full of low blue dots** — many players had star-*looking*
   profiles and flopped. A strong profile is **not sufficient**.

"Must have it, but having it guarantees nothing" is exactly what a ~0.4 correlation looks
like in picture form.

The **side-by-side** version shows two ways to build the index:
- **Equal-weight (r = 0.33):** every stat weighted the same (a plain average)
- **Regression-weighted (r = 0.43):** the computer learns which stats matter more

The right panel is tighter, but the superstars still float far above the trend line in
both — no cleverness with college stats lifts you past the wall.

---

## How the computer "learns which stats matter"

It's the A-level **line of best fit**, extended.

A line of best fit is `y = slope × x + intercept`, chosen to sit as close to all the dots
as possible (it minimises the total *squared* vertical gap — "least squares").

With five stats instead of one, you get five slopes (called **weights**) plus a baseline:

```
predicted VORP = 6.61
              + 0.162 × BPM
              + 0.226 × scoring (porpag)
              - 0.030 × offensive rating (adjoe)
              + 0.046 × shot volume (usg)
              - 0.254 × age at draft
```

The computer picks the weights that make predictions closest to reality across ~680 past
players. A stat that helps gets a big weight; one that doesn't gets ~0.

What it decided, and why it's smart:
- **Scoring (+0.23)** and **youth (−0.25, big negative = younger is better)** are the
  strongest clues.
- **Shot volume (+0.05)** ≈ ignored — *sounds* impressive but adds nothing once you know
  scoring and quality.
- **Offensive rating (−0.03)** ≈ deleted — it **overlaps** with BPM and scoring (same red
  family from the checkerboard), so the model refuses to pay twice for the same clue.

**Overfitting caveat:** a computer tuning weights freely can "memorise" flukes of these
particular players (like memorising last year's exam). To check honestly, we hide some
players, train on the rest, then test on the hidden ones. The **r = 0.43 is measured on
players it never saw** — real skill, not memorisation.

---

## Watching one prediction get made

The recipe is just: **baseline 6.61, then each stat × its weight, added up.**

### Stephen Curry

| | raw stat | × weight | = nudge |
|---|---|---|---|
| baseline | | | +6.61 |
| BPM | 12.31 | 0.162 | +2.00 |
| scoring | 6.53 | 0.226 | +1.48 |
| off. rating | 140.82 | −0.030 | −4.28 |
| shot volume | 37.50 | 0.046 | +1.71 |
| age (21.7) | 21.69 | −0.254 | −5.51 |
| | | **Predicted** | **2.0** |
| | | **Actual** | **8.0** |

The big negative nudges look scary only because age (~21) and offensive rating (~140) are
large raw numbers times small weights — the high baseline (6.61) absorbs them. It balances
to a sensible 2.0.

### Predicted vs actual, four players

| Player | Predicted | Actual | What happened |
|---|---|---|---|
| Stephen Curry | 2.0 | **8.0** | Massive under-shot — his shooting genius isn't in the box score |
| Anthony Davis | 2.6 | 5.5 | Under-shot again despite the model liking him |
| Isaiah Thomas | 0.3 | 4.0 | Written off (old, pick #60) — became a star anyway |
| James Wiseman | 1.8 | **−0.2** | Great-looking profile, top-2 pick — flopped. A false positive |

Every prediction is off, and **the misses have no pattern** the model could fix: it
under-shoots stars, over-shoots busts, penalises the wrong older players. If the errors
had a pattern, the weights would already capture it. They don't, so they can't.

---

## The bottom line

r = 0.43 means the recipe gets the **general direction** right (better stats → higher
predictions on average) but is routinely wrong by 2–4 VORP for any **individual** player —
because what separates a 2.0 player from an 8.0 player (Curry's shooting, Wiseman's
fragility, health, situation, coaching) was never in the college numbers.

**College stats reliably tell you who *can't* clear the floor. They cannot tell you who
*will* hit the ceiling.** The computer did its job perfectly — the job just has a ceiling.

## Why the position splits matter

Pooling point guards and centers can create misleading relationships because their
normal college responsibilities are different. The position analysis therefore reports
PG, SG, SF, PF, and C correlations separately, plus a pooled result and a
within-position-controlled result.

For example, the Possession Creation Index correlation with surplus value is strongest
for small forwards (`r = 0.245`) and weakest for centers (`r = 0.075`). The Granular
Prospect Index remains positive at every position, ranging from `r = 0.260` for power
forwards to `r = 0.466` for point guards. The scoring-profile score changes direction:
slightly positive for guards, approximately zero for small forwards, and slightly
negative for power forwards and centers.

Those subgroup values are clearer, but not automatically more precise: each position has
fewer players and therefore wider uncertainty. Use
`data/position_feature_correlations.csv` to see sample size, confidence interval,
pick-adjusted partial correlation, and multiple-testing-adjusted q-value for every result.

The historical primary position is the player's first recorded NBA position. It is a
descriptive grouping and is never passed into the leakage-safe prediction model.

---

*Generated alongside `new.py`, `scratchpad_composite_plot.py`, and
`scratchpad_composite_compare.py`.*
