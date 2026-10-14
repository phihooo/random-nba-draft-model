import { useState } from "react";

const C = {
  bg: "#0f172a",
  card: "#1e293b",
  border: "#334155",
  text: "#f1f5f9",
  muted: "#94a3b8",
  dim: "#64748b",
  orange: "#f97316",
  blue: "#3b82f6",
  green: "#22c55e",
  purple: "#a855f7",
  cyan: "#06b6d4",
  red: "#ef4444",
  yellow: "#eab308",
};

const phases = [
  {
    id: "p0",
    title: "Phase 0 — Learn the Landscape",
    when: "Now → Berkeley move-in (3 weeks)",
    hours: "~90 min, twice per week",
    color: C.orange,
    goal: "Understand how the existing metrics work conceptually. You're not building anything yet — you're reading, watching, and developing intuition.",
    tasks: [
      {
        name: "Read: How RAPM works",
        detail: "Regularised Adjusted Plus-Minus is the foundation of every modern basketball metric. It asks: 'How much does the score change when this player is on the court, adjusted for teammates and opponents?' Read Jeremias Engelmann's original paper or the Thinking Basketball summary. You don't need to understand the matrix algebra yet — understand the concept: isolating a player's impact from their context.",
        time: "2 sessions",
      },
      {
        name: "Read: How DARKO, LEBRON, and RAPTOR differ",
        detail: "DARKO (by Kostya Medvedovsky): Bayesian model that projects future performance using historical player comparisons and box score stats. Publicly available projections at apanacea.com.\n\nLEBRON (by BBall Index): combines box-score and tracking data with luck-adjusted metrics. Proprietary but outputs are published.\n\nRAPTOR (by FiveThirtyEight, now discontinued): blended box-score and player tracking with a Bayesian prior based on player type. Read the methodology writeup on FiveThirtyEight.\n\nYou don't need to replicate these. Understand: what inputs do they use? What are they trying to predict? Where do they disagree?",
        time: "3 sessions",
      },
      {
        name: "Explore data sources",
        detail: "Bookmark these — you'll use them in Phase 1:\n\n• Basketball Reference (bbref.com): college and NBA stats, draft history, advanced stats (VORP, BPM, WS). Free.\n• Stathead (stathead.com): advanced queries. Paid but has a free tier.\n• NBA API (via nba_api Python package): play-by-play, shot charts, tracking data.\n• Barttorvik (barttorvik.com): best source for college basketball advanced stats. Has transfer portal data too.\n• Cleaning the Glass (cleaningtheglass.com): excellent context-adjusted NBA stats. Paid, worth it later.\n• Tank Athon or Ringer mock drafts: for historical consensus draft positions.",
        time: "1 session",
      },
      {
        name: "Watch: Thinking Basketball YouTube channel",
        detail: "Ben Taylor's channel is the gold standard for understanding basketball analytics intuitively. His 'Greatest Peaks' series demonstrates exactly the kind of thinking your model will try to quantify. Watch 3-4 episodes. Notice how he separates 'box score production' from 'actual impact' — that distinction is what your project is about.",
        time: "Background viewing",
      },
    ],
  },
  {
    id: "p1",
    title: "Phase 1 — Data Collection & Exploration",
    when: "Berkeley Weeks 1-4 (August-September)",
    hours: "~1 hour/day, 5 days/week",
    color: C.blue,
    goal: "Collect historical draft data and NBA career stats. Build your first dataset. Do basic exploratory analysis. This phase runs in parallel with Data 8, which will teach you the exact Python skills you need.",
    tasks: [
      {
        name: "Scrape/download draft history (2000-2022)",
        detail: "From Basketball Reference: for each draft from 2000 to 2022, collect every pick's name, position, draft number, college, age, and key college stats.\n\nWhy 2000-2022: pre-2000 data has different statistical context. Post-2022 players haven't had enough NBA time to evaluate.\n\nTools: pandas + basketball_reference_web_scraper or manual CSV downloads from bbref. Data 8 Week 1-2 covers exactly this.",
        time: "Week 1-2",
      },
      {
        name: "Collect NBA career stats for each drafted player",
        detail: "For each player in your draft dataset, pull their first 4 seasons of NBA stats: games played, minutes, VORP, BPM, Win Shares, Win Shares per 48.\n\nKey decision: use first 4 seasons as the evaluation window. This gives enough time for development while keeping the scope manageable.\n\nAlso record: did they get a second contract? How big was it relative to the cap?",
        time: "Week 2-3",
      },
      {
        name: "Collect college stats from Barttorvik",
        detail: "For each drafted player, pull their final college season stats. Focus on YOUR features — the possession-economics stats:\n\n• Turnover rate (TOV%)\n• Steal rate (STL%)\n• Offensive rebound rate (ORB%)\n• Defensive rebound rate (DRB%)\n• Assist rate (AST%)\n• Assist-to-turnover ratio\n• Free throw rate (FTr)\n• Usage rate (USG%)\n• Block rate (BLK%)\n\nAlso collect the conventional features for comparison: PPG, PER, TS%, etc.",
        time: "Week 3-4",
      },
      {
        name: "Build the baseline: 'expected value by draft slot'",
        detail: "Calculate the historical average VORP/BPM/WS for each draft position across your 22 drafts. Plot it. You'll see a clear curve: pick 1 averages X VORP, pick 10 averages Y, pick 30 averages Z. This curve is 'what the market expects.' Any player above the curve was undervalued. Below = overvalued.\n\nThis plot is your project's anchor visual. It will appear in every presentation.",
        time: "Week 4",
      },
      {
        name: "First exploration: who were the biggest surprises?",
        detail: "Sort your data by surplus value (actual VORP minus expected VORP for draft slot). Who are the most undervalued players in your dataset? Jokić at 41? Kawhi at 15? Draymond at 35?\n\nNow look at their college possession stats. Do you see a pattern? This is where your hypothesis starts forming with data behind it, not just intuition.",
        time: "Week 4",
      },
    ],
  },
  {
    id: "p2",
    title: "Phase 2 — Feature Engineering & Hypothesis Testing",
    when: "Berkeley Weeks 5-9 (October-early November)",
    hours: "~1 hour/day",
    color: C.green,
    goal: "Build your feature set, test your possession-economics hypothesis, and start basic modelling. Data 8 will have covered regression by this point.",
    tasks: [
      {
        name: "Create your 'Possession Creation Index'",
        detail: "This is your signature feature — a composite metric that captures your basketball philosophy. Combine:\n\n• Steal rate (creating possessions via defense)\n• Offensive rebound rate (creating possessions via second chances)\n• Inverse turnover rate (preserving possessions)\n• Assist rate (converting possessions into quality shots)\n\nWeight them however your intuition says, then test whether the composite predicts surplus value. This is YOUR metric — it encodes YOUR theory of basketball value.",
        time: "Week 5-6",
      },
      {
        name: "Test the core hypothesis",
        detail: "Split your features into two groups:\n\nGroup A (conventional): PPG, PER, TS%, USG%\nGroup B (possession-economics): your Possession Creation Index components\n\nRun separate correlations with surplus value. Which group predicts better? If Group B wins, your hypothesis is supported. If Group A wins, you've learned something important and can refine.\n\nData 8 covers correlation and regression in Weeks 5-8 — your coursework will directly teach you how to do this.",
        time: "Week 6-7",
      },
      {
        name: "Add context features",
        detail: "Some features that might improve predictions:\n\n• Age at draft (younger = more upside)\n• Conference strength (Big 12 vs Patriot League)\n• Role on college team (was he the #1 option or #3?)\n• Physical measurements from the combine (wingspan, standing reach)\n• Whether they were a transfer\n\nAdd these one at a time and see if they improve your model.",
        time: "Week 7-8",
      },
      {
        name: "Build Model v1: Linear Regression",
        detail: "Predict surplus value using your features. Start with just your Possession Creation Index. Then add features one by one. Track R² (how much variance your model explains). Even a modest R² (0.15-0.25) on draft surplus value would be meaningful — draft prediction is inherently noisy.\n\nCritically: use train/test split or cross-validation (Data 8 covers this). Predict on drafts your model hasn't seen. This is the backtesting step that makes it credible.",
        time: "Week 8-9",
      },
    ],
  },
  {
    id: "p3",
    title: "Phase 3 — Refinement & Storytelling",
    when: "Berkeley Weeks 10-15 (November-December)",
    hours: "~1 hour/day",
    color: C.purple,
    goal: "Improve the model, build visualisations, write up findings, and polish the GitHub repo. By December, this should be presentable.",
    tasks: [
      {
        name: "Backtest against recent drafts",
        detail: "Hold out the 2019-2022 drafts entirely. Train on 2000-2018. Predict surplus value for 2019-2022 picks. Which players does your model flag as undervalued? Do those predictions match reality?\n\nThis is the money slide. 'My model would have told you Tyrese Haliburton at pick 12 was undervalued' (or whoever your model surfaces). Concrete, verifiable, compelling.",
        time: "Week 10-11",
      },
      {
        name: "The key finding: what does the market underweight?",
        detail: "This is the stat arb punchline. Compare the coefficients in your model:\n\n• If steal rate and turnover rate have high coefficients but the draft market doesn't reflect them → the market systematically underweights possession creation\n• If your Possession Creation Index adds significant predictive power beyond conventional stats → you've found a market inefficiency\n\nState the finding clearly. Even a negative result ('the market is actually efficient on possession stats') is interesting and publishable.",
        time: "Week 11-12",
      },
      {
        name: "Build 3-4 key visualisations",
        detail: "1. Draft slot vs actual VORP curve (with your model's predictions overlaid)\n2. Surplus value by Possession Creation Index quartile (do high-PCI players outperform?)\n3. Feature importance chart (what predicts surplus value most?)\n4. Your model's top 'steals' for a recent draft vs where they actually went\n\nUse matplotlib or seaborn (both covered in Data 8).",
        time: "Week 12-13",
      },
      {
        name: "Write the README and polish GitHub",
        detail: "Your GitHub README should read like a short research paper:\n\n1. Question (one sentence)\n2. Data (sources, scope, size)\n3. Method (features, model, backtesting approach)\n4. Key finding (one paragraph)\n5. Visualisations (embedded)\n6. Limitations and next steps\n\nThis README is what a recruiter or Traders at Berkeley interviewer will actually read. Make it clean, specific, and honest about limitations.",
        time: "Week 14-15",
      },
    ],
  },
  {
    id: "p4",
    title: "Phase 4 — Extensions (Spring & Beyond)",
    when: "Spring Year 1 onwards",
    hours: "As interest dictates",
    color: C.cyan,
    goal: "Once v1 is complete, these are the directions you can take it as your skills grow through CS 61A, Stat 134, and CS 189.",
    tasks: [
      {
        name: "Add ML models (after CS 61A / CS 189)",
        detail: "Replace linear regression with Random Forest, Gradient Boosting, or eventually neural networks. Compare: does added model complexity actually improve prediction, or does simple regression capture most of the signal? (Often it does — and knowing that is itself a valuable finding.)",
        time: "Spring",
      },
      {
        name: "Build the Option C extension: outcome distributions",
        detail: "Remember Option C from our earlier discussion? Instead of predicting average surplus value, predict the DISTRIBUTION. 'This player has a 25% chance of All-Star, 40% starter, 35% bust.' This requires quantile regression or mixture models. Connects directly to options pricing and tail risk — deep quant territory.",
        time: "Year 2",
      },
      {
        name: "Apply to the 2027 draft in real-time",
        detail: "Run your model on the incoming 2027 draft class. Publish predictions BEFORE the draft. Then track results. A public, time-stamped prediction that you can later verify is the most credible form of backtesting.",
        time: "Spring Year 1 → June 2027",
      },
      {
        name: "Connect to actual betting markets",
        detail: "Draft position betting markets exist. If your model identifies a player the market has at pick 15 but your model values at pick 7, that's a quantifiable edge. Calculate: if you had bet on your model's top 5 undervalued picks each year, what would your return have been? This is the full stat arb loop.",
        time: "Year 2",
      },
      {
        name: "Submit to the Sloan Sports Analytics Conference",
        detail: "MIT Sloan Sports Analytics Conference accepts undergraduate research. A polished version of this project — especially if your 2027 draft predictions prove accurate — is a legitimate submission. Having a Sloan paper as a freshman/sophomore is a massive differentiator.",
        time: "Year 2",
      },
    ],
  },
];

function TaskCard({ task, isOpen, onToggle }) {
  return (
    <div
      onClick={onToggle}
      style={{
        background: isOpen ? `${C.border}88` : `${C.border}33`,
        borderRadius: 6,
        padding: "10px 14px",
        marginBottom: 5,
        cursor: "pointer",
        transition: "background 0.15s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span style={{ fontWeight: 600, fontSize: 13, color: C.text, lineHeight: 1.4, flex: 1 }}>
          {task.name}
        </span>
        {task.time && (
          <span style={{ fontSize: 11, color: C.dim, whiteSpace: "nowrap", marginLeft: 8, fontFamily: "system-ui, sans-serif" }}>
            {task.time}
          </span>
        )}
      </div>
      {isOpen && (
        <div style={{ marginTop: 8, fontSize: 12.5, color: C.muted, lineHeight: 1.65, whiteSpace: "pre-line" }}>
          {task.detail}
        </div>
      )}
    </div>
  );
}

export default function ProjectRoadmap() {
  const [activePhase, setActivePhase] = useState("p0");
  const [openTasks, setOpenTasks] = useState(new Set());

  const toggleTask = (key) => {
    setOpenTasks((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const active = phases.find((p) => p.id === activePhase);

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", background: C.bg, color: C.text, minHeight: "100vh", padding: "24px 20px" }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>
        <h1 style={{ fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: "-0.02em" }}>
          NBA Draft Surplus Value Model
        </h1>
        <p style={{ color: C.muted, fontSize: 13, margin: "6px 0 4px", lineHeight: 1.5 }}>
          "Does the NBA draft market systematically undervalue possession-creation players?"
        </p>
        <p style={{ color: C.dim, fontSize: 12, margin: "0 0 20px" }}>
          Target variable: VORP minus expected VORP for draft slot · Feature angle: possession economics · Tap tasks for details
        </p>

        <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 20 }}>
          {phases.map((p) => (
            <button
              key={p.id}
              onClick={() => setActivePhase(p.id)}
              style={{
                padding: "7px 12px",
                borderRadius: 7,
                border: "none",
                cursor: "pointer",
                background: activePhase === p.id ? p.color : C.card,
                color: activePhase === p.id ? "#000" : C.text,
                fontSize: 12,
                fontWeight: 600,
                transition: "all 0.15s",
              }}
            >
              {p.title.split("—")[0].trim()}
            </button>
          ))}
        </div>

        {active && (
          <div
            style={{
              background: C.card,
              borderRadius: 10,
              padding: 16,
              border: `1px solid ${C.border}`,
              marginBottom: 16,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
              <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0, color: active.color }}>
                {active.title}
              </h2>
            </div>
            <div style={{ display: "flex", gap: 12, marginBottom: 12, fontSize: 12, color: C.muted }}>
              <span>📅 {active.when}</span>
              <span>⏱ {active.hours}</span>
            </div>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
              {active.goal}
            </p>
            {active.tasks.map((task, i) => {
              const key = `${active.id}-${i}`;
              return (
                <TaskCard
                  key={key}
                  task={task}
                  isOpen={openTasks.has(key)}
                  onToggle={() => toggleTask(key)}
                />
              );
            })}
          </div>
        )}

        <div
          style={{
            background: C.card,
            borderRadius: 10,
            padding: 16,
            border: `1px solid ${C.border}`,
            fontSize: 13,
            lineHeight: 1.7,
          }}
        >
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: "0 0 10px" }}>
            How This Sounds in a Quant Interview
          </h3>
          <p style={{ margin: "0 0 8px", color: C.text }}>
            <em>"I built a model predicting NBA draft surplus value using possession-based features — turnover rate, steal rate, offensive rebound rate. My hypothesis was that the draft market systematically underweights possession creation relative to scoring volume. I trained on 2000-2018 drafts and backtested on 2019-2022. The model identified [X] out of [Y] top surplus-value players in the holdout set, and the possession-creation features added [Z]% predictive power beyond conventional box score stats."</em>
          </p>
          <p style={{ margin: 0, color: C.muted }}>
            That paragraph demonstrates: hypothesis formation, feature engineering, train/test discipline, backtesting methodology, and the ability to find signal in noisy data. These are exactly the skills quant firms interview for.
          </p>
        </div>
      </div>
    </div>
  );
}
