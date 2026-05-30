# Ligue 1 Match Outcome Predictor — Predictive Machine Learning Engine

An end-to-end data science pipeline developed to predict match outcomes for the French Ligue 1 football season. 

This engine ingest over a decade of historical football data (2012–2025) across player appearances, market valuations, match events, and team statistics to solve a challenging three-class classification problem: predicting a **Home Win (`1`)**, a **Draw (`0`)**, or an **Away Win (`-1`)**.

##  Repository Structure

* **`ligue1_prediction_Yasmine.ipynb`**: The primary data science notebook containing the complete exploratory data analysis (EDA), multi-table aggregation pipelines, feature engineering, model training, and performance evaluation.
* **`football csv files/`**: The structured data directory containing the 8 source relational datasets:
  * `matchs_2013_2024.csv`: Core training match history (~4,700 games with terminal outcomes).
  * `match_2025.csv`: The target test matching fixture pipeline for outcome predictions.
  * `clubs_fr.csv`: High-level squad snapshots (average age, squad size, stadium capacity).
  * `player_valuation_before_season.csv`: Historical player market values over time.
  * `player_appearance.csv`: Match-by-match individual performance metrics (minutes, cards, goals).
  * `game_events_before2025.csv`: Granular in-game actions (substitutions, card contexts, assists).
  * `game_lineups.csv`: Strategic starting lineups for historical matches.
  * `sample_results.csv`: Formatting blueprint for output test predictions.

---

##  Core Engineering Challenges & Constraints

### 1. The Future Lineup Blindness Constraint
* **The Problem:** In real-world deployment, match lineups are never finalized or announced until roughly an hour before kickoff. Therefore, predicting matchups days or weeks in advance means a machine learning model cannot rely heavily on active individual player line sheets.
* **The Solution:** Engineered aggregated club-level signals. Individual player histories, market valuations, and historic match counts are rolled up dynamically into macro team indicators, transforming granular player metrics into consistent, stable organizational features.

### 2. High-Stakes Financial Scaling
* Utilized raw temporal data from `player_valuation_before_season.csv` to construct club financial profiles. 
* By normalizing market values into rolling squad-wealth ratios, the model successfully captures systemic power dynamics (e.g., the widening resource disparities between top-tier title contenders and newly promoted clubs) to anchor predictive baselines.

---

## 🛠️ Feature Engineering Pipeline

Because football metrics suffer from heavy noise, the project focuses on robust, rolling aggregate features built from the ground up:

* **Rolling Team Performance Form:** Computes exponentially weighted or fixed-window moving results over historical matchdays to capture momentum, home-field advantage multipliers, and away fatigue.
* **Offensive & Defensive Core Power:** Aggregates historic match events (`game_events_before2025.csv`) and goal/assist statistics to establish time-decayed scoring and defensive resilience metrics for each club.
* **Squad Maturity Metrics:** Leverages squad age brackets and national team selection density flags to model team depth and structural performance under high-pressure gameweeks.

---

##  Performance & Baseline Benchmarks

* **Target Evaluation:** Outperforming a naive baseline (e.g., always predicting home wins or a random uniform distribution).
* **Predictive Sweet Spot:** Achieves an accuracy velocity of **~52% – 54%**. 
* **Domain Reality:** Sports analytics domains consider 52%+ accuracy on multi-class match results highly competitive. Commercial betting algorithms heavily reliant on real-time injury logs and localized variables generally peak around 55%–60%, validating the predictive power of this pure historical model.

##  Tech Stack & Dependencies
* **Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Methodologies:** Multi-table relational joins, data aggregation pipelines, time-series feature windows, Multi-class Classification.
