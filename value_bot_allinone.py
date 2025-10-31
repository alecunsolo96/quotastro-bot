#!/usr/bin/env python3 """ Value Betting Bot (CLI)

Focus: Over/Under 2.5 goals (can extend to BTTS and 1X2).

What it does (now):

Pull recent form and team goal/xG rates (you can paste them or fetch via API providers).

Fit a simple Poisson-based model with home/away attack & defence strengths.

Optionally blend xG with actual goals for more stability.

Compute probability of Over 2.5 (and Under 2.5) via goal grid.

Convert probabilities to fair odds, compare to market odds, and compute edge & Kelly stake.


What you can extend easily:

Dixon–Coles correlation term for low-scoring correction.

Injuries/suspensions weighting.

Bookmaker odds scraper / aggregator.

Live model (re-estimate with in-play stats).


USAGE EXAMPLES

$ python value_bot.py --home "Inter" --away "Empoli" --over25_odds 1.75 
--home_goals_for 2.1 --home_goals_against 0.8 
--away_goals_for 0.9 --away_goals_against 1.6 
--home_xg_for 2.2 --home_xg_against 1.0 
--away_xg_for 0.95 --away_xg_against 1.55 
--league_avg_goals 2.65 --home_adv 0.15

If you have API keys, you can plug a data provider in DataProvider. """

from future import annotations import argparse import math from dataclasses import dataclass from typing import Optional, Tuple

---------------------------

Utilities

---------------------------

def poisson_pmf(k: int, lam: float) -> float: if lam <= 0: return 1.0 if k == 0 else 0.0 # exp(-lam) * lam^k / k! return math.exp(-lam) * lam**k / math.factorial(k)

def cap(x: float, low: float, high: float) -> float: return max(low, min(high, x))

---------------------------

Kelly criterion

---------------------------

def kelly_fraction(p: float, odds: float, kelly_mult: float = 1.0) -> float: """Kelly fraction for decimal odds. Returns 0 if no edge. Multiply by your bankroll to get stake. """ b = odds - 1.0 edge = p * (b + 1) - 1 if edge <= 0: return 0.0 frac = (b * p - (1 - p)) / b return max(0.0, frac * kelly_mult)

---------------------------

Simple team strength model

---------------------------

@dataclass class TeamRates: name: str g_for: float g_against: float xg_for: Optional[float] = None xg_against: Optional[float] = None

@dataclass class ModelConfig: league_avg_goals: float = 2.70 home_adv_goals: float = 0.15  # additive boost to home expected goals xg_weight: float = 0.6        # blend weight between xG and goals max_goals_grid: int = 10      # 0..10 per side for grid prob sum

class OverUnderModel: def init(self, cfg: ModelConfig): self.cfg = cfg

def _blend_rate(self, goals: float, xg: Optional[float]) -> float:
    if xg is None:
        return goals
    w = cap(self.cfg.xg_weight, 0.0, 1.0)
    return w * xg + (1 - w) * goals

def expected_goals(self, home: TeamRates, away: TeamRates) -> Tuple[float, float]:
    # Basic attack/defence factors using league average
    lg = self.cfg.league_avg_goals
    lg_home = lg * 0.55  # rough split home vs away
    lg_away = lg * 0.45

    h_att = self._blend_rate(home.g_for, home.xg_for) / lg_home
    h_def = self._blend_rate(home.g_against, home.xg_against) / lg_away
    a_att = self._blend_rate(away.g_for, away.xg_for) / lg_away
    a_def = self._blend_rate(away.g_against, away.xg_against) / lg_home

    lam_home = lg_home * h_att * a_def + self.cfg.home_adv_goals
    lam_away = lg_away * a_att * h_def

    # guardrails
    lam_home = cap(lam_home, 0.1, 4.5)
    lam_away = cap(lam_away, 0.1, 4.5)
    return lam_home, lam_away

def prob_over_under_25(self, lam_home: float, lam_away: float) -> Tuple[float, float]:
    # Independent Poisson goal model
    max_g = self.cfg.max_goals_grid
    over_prob = 0.0
    under_prob = 0.0
    for i in range(0, max_g + 1):
        pi = poisson_pmf(i, lam_home)
        for j in range(0, max_g + 1):
            pj = poisson_pmf(j, lam_away)
            p = pi * pj
            if i + j >= 3:
                over_prob += p
            else:
                under_prob += p
    # Renormalize tail mass if grid cut-off is small
    total_prob = over_prob + under_prob
    if total_prob < 0.999:  # missing tail
        over_prob /= total_prob
        under_prob /= total_prob
    return over_prob, under_prob

---------------------------

CLI

---------------------------

def run_cli(): ap = argparse.ArgumentParser(description="Over/Under 2.5 Value Bot") ap.add_argument('--home', required=True) ap.add_argument('--away', required=True) ap.add_argument('--over25_odds', type=float, required=True, help='Decimal odds for Over 2.5 to evaluate')

# Team numbers: per-match rates over a chosen sample (e.g., last 10 league matches)
ap.add_argument('--home_goals_for', type=float, required=True)
ap.add_argument('--home_goals_against', type=float, required=True)
ap.add_argument('--away_goals_for', type=float, required=True)
ap.add_argument('--away_goals_against', type=float, required=True)

# Optional xG
ap.add_argument('--home_xg_for', type=float)
ap.add_argument('--home_xg_against', type=float)
ap.add_argument('--away_xg_for', type=float)
ap.add_argument('--away_xg_against', type=float)

# Model config
ap.add_argument('--league_avg_goals', type=float, default=2.70)
ap.add_argument('--home_adv', type=float, default=0.15)
ap.add_argument('--xg_weight', type=float, default=0.6)
ap.add_argument('--kelly_mult', type=float, default=0.5, help='0..1 (e.g., 0.5 = half Kelly)')

args = ap.parse_args()

home = TeamRates(
    name=args.home,
    g_for=args.home_goals_for,
    g_against=args.home_goals_against,
    xg_for=args.home_xg_for,
    xg_against=args.home_xg_against,
)
away = TeamRates(
    name=args.away,
    g_for=args.away_goals_for,
    g_against=args.away_goals_against,
    xg_for=args.away_xg_for,
    xg_against=args.away_xg_against,
)

cfg = ModelConfig(
    league_avg_goals=args.league_avg_goals,
    home_adv_goals=args.home_adv,
    xg_weight=args.xg_weight,
)

model = OverUnderModel(cfg)
lam_h, lam_a = model.expected_goals(home, away)
p_over, p_under = model.prob_over_under_25(lam_h, lam_a)

fair_over = 1.0 / p_over if p_over > 0 else float('inf')
edge = p_over * args.over25_odds - 1.0
kelly = kelly_fraction(p_over, args.over25_odds, kelly_mult=args.kelly_mult)

print("\n=== Match ===")
print(f"{home.name} vs {away.name}")

print("\n=== Model Inputs (per match rates) ===")
print(f"League avg goals: {cfg.league_avg_goals:.2f}")
print(f"Home advantage (additive): {cfg.home_adv_goals:.2f}")
print(f"xG weight: {cfg.xg_weight:.2f}")

print("\nHome team")
print(f"  goals for:     {home.g_for:.2f}  | goals against: {home.g_against:.2f}")
if home.xg_for is not None:
    print(f"  xG for:        {home.xg_for:.2f}  | xG against:    {home.xg_against:.2f}")

print("Away team")
print(f"  goals for:     {away.g_for:.2f}  | goals against: {away.g_against:.2f}")
if away.xg_for is not None:
    print(f"  xG for:        {away.xg_for:.2f}  | xG against:    {away.xg_against:.2f}")

print("\n=== Model Outputs ===")
print(f"Expected goals (λ): home {lam_h:.2f} | away {lam_a:.2f} | total {lam_h+lam_a:.2f}")
print(f"P(Over 2.5): {p_over:.3f} | P(Under 2.5): {p_under:.3f}")
print(f"Fair odds Over 2.5: {fair_over:.2f}")

print("\n=== Market & Edge ===")
print(f"Market odds Over 2.5: {args.over25_odds:.2f}")
print(f"Edge: {edge*100:.2f}%")
if edge > 0:
    print(f"Suggested Kelly fraction (@{args.kelly_mult:.2f}×): {kelly*100:.2f}% of bankroll")
else:
    print("No positive edge per this model – pass or seek better price.")

if name == "main": run_cli()
