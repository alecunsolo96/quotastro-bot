#!/usr/bin/env python3
"""
AI Betting Bot (Poisson + xG Under/Over Model)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------- Poisson PMF ----------
def poisson_pmf(k: int, lam: float) -> float:
    return (lam**k * math.exp(-lam)) / math.factorial(k)

# ---------- Helper ----------
def cap(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))

# ---------- Team Data ----------
@dataclass
class TeamStats:
    goals_for: float
    goals_against: float
    xg_for: float
    xg_against: float

# ---------- Model ----------
def expected_goals(home: TeamStats, away: TeamStats, league_avg: float, home_adv: float = 0.15):
    home_rate = (home.goals_for + away.goals_against + home.xg_for + away.xg_against)/4 + home_adv
    away_rate = (away.goals_for + home.goals_against + away.xg_for + home.xg_against)/4
    home_rate = cap(home_rate, 0.3, 3.5)
    away_rate = cap(away_rate, 0.1, 3.0)
    return home_rate, away_rate

def prob_over_under_25(home_rate: float, away_rate: float):
    probs = []
    for hg in range(0, 10):
        for ag in range(0, 10):
            p = poisson_pmf(hg, home_rate) * poisson_pmf(ag, away_rate)
            probs.append((hg+ag, p))
    over = sum(p for g,p in probs if g >= 3)
    under = sum(p for g,p in probs if g <= 2)
    return over, under

# ---------- Entry function ----------
def get_prediction(home: TeamStats, away: TeamStats, league_avg: float = 2.6):
    home_rate, away_rate = expected_goals(home, away, league_avg)
    over, under = prob_over_under_25(home_rate, away_rate)
    return {
        "expected_home_goals": round(home_rate,2),
        "expected_away_goals": round(away_rate,2),
        "prob_over_2_5": round(over,3),
        "prob_under_2_5": round(under,3),
        "fair_odds_over": round(1/over,2) if over > 0 else None,
        "fair_odds_under": round(1/under,2) if under > 0 else None,
    }

# ---------- Test Run ----------
if __name__ == "__main__":
    home = TeamStats(2.1, 1.0, 2.2, 1.1)
    away = TeamStats(0.9, 1.4, 1.0, 1.6)
    
    result = get_prediction(home, away)
    print(result)
