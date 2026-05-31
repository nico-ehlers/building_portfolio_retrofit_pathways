# Retrofit Priority Algorithm

Multi-Criteria Decision Making (MCDM) algorithm for prioritising building retrofit measures in a municipal or institutional building portfolio.

**Published in:**
> Ehlers, N. et al. (2025). *A multi-criteria approach for prioritising building envelope and HVAC retrofits in a municipal building portfolio.* Energy and Buildings.

---

## Overview

The algorithm assigns a numerical priority score to every building-scenario combination and uses that score to generate a multi-year retrofit schedule. It proceeds in four steps:

```
Raw portfolio data
       │
       ▼
1. Preprocessing       – encode qualitative fields, impute missing values,
                         Quantile-Uniform scale all criteria
       │
       ▼
2. MCDM Scoring        – weighted linear aggregation of scaled criteria
                         → total score per building-scenario
       │
       ▼
3. Scenario Selection  – reduce to one scenario per building via rule-based
                         filtering (combined / construction-only / HVAC-only)
       │
       ▼
4. Retrofit Schedule   – greedy bin-packing with look-ahead assigns a
                         retrofit year to every building
```

---

## Method

### Scaling

All criteria are scaled to [0, 1] using the **Quantile (Rank) Transformer** (uniform output distribution). This approach is robust against outliers, ensuring that the most frequent values are spread across the full range rather than compressed at the extremes.

For criteria where a *lower* physical value is better (investment cost, embodied GHG, lifecycle cost), the scaled value is **inverted** (1 − scaled).

### Weighting & Scalarisation

Criteria are grouped into thematic clusters. A two-level weighting scheme is applied:

```
final_weight(criterion) = cluster_weight × weight_within_cluster
```

The total MCDM score for each building-scenario is the weighted linear sum:

```
score = Σ  f_i × x_i_scaled
```

where `f_i` is the weight for criterion `i` and `x_i_scaled` is its scaled value.

### Predefined weighting scenarios

| Scenario | Priority | NFA-dep. | Materials | Performance |
|---|---|---|---|---|
| `weighted_stmb` | 30 % | 20 % | 10 % | 40 % |
| `balanced` | 25 % | 25 % | 25 % | 25 % |
| `extreme` | 30 % | 10 % | 0 % | 60 % |
| `simple_rsl` | 100 % | – | – | – |

### Scenario Selection

Each building may have multiple retrofit scenarios (envelope-only, HVAC-only, combined). After scoring, the following rules select exactly one scenario per building:

1. **Combined on top** → keep the combined scenario.
2. **Construction-only on top** → find the matching combined scenario, copy its performance metrics (energy/GHG/cost savings) into the best HVAC-only row to preserve cost data, and keep that row. Falls back to the construction-only row if no match exists.
3. **HVAC-only on top** → keep the combined scenario whose name contains the top HVAC-only name; fall back to the best combined, or the HVAC-only row itself.

### Retrofit Scheduling

A greedy bin-packing algorithm distributes buildings across years. The annual capacity target is either:
- `total_nfa × quotient`  *(quotient mode, default 3 %/a)*
- `total_nfa / (end_year − start_year + 1)`  *(end-year mode)*

Buildings are added in score order. When a building would exceed the annual capacity, a **look-ahead pass** scans all remaining buildings to fill the remaining space within the allowed threshold before closing the year.

---

## Criteria Reference

| Key | Label | Description | Direction |
|---|---|---|---|
| `measure` | u5 | Retrofit scope (combined = 1.0) | ↑ higher is better |
| `hvac_rsl` | t1 | Remaining HVAC service life | ↑ short RSL = urgent |
| `envbpot` | t2 | Envelope improvement potential | ↑ higher is better |
| `energy_red` | t3 | Specific final energy savings [kWh/(m²·a)] | ↑ |
| `penrt_red` | t4 | Specific non-renewable primary energy savings [kWh/(m²·a)] | ↑ |
| `gwp_red` | t5 | Specific GHG savings – operational [kgCO₂eq/(m²·a)] | ↑ |
| `cost_red` | t6 | Specific operational cost savings [€/(m²·a)] | ↑ |
| `cost_energy_red` | t7 | Cost efficiency: energy savings per investment [kWh/(€·m²·a)] | ↑ |
| `invest_cost` | t8 | Total investment cost [€] | ↓ inverted |
| `pv_energy` | t9 | PV energy potential [kWh/a] | ↑ |
| `penrt_materials` | t10 | Embodied non-renewable primary energy [kWh/m²] | ↓ inverted |
| `gwp_materials` | t11 | Embodied GHG of materials [kgCO₂eq/m²] | ↓ inverted |
| `lcc_materials` | t12 | Lifecycle cost of materials [€/m²] | ↓ inverted |
| `pet_red` | t13 | Specific total primary energy savings [kWh/(m²·a)] | ↑ |

---

## Input Data Format

The function `preprocess_portfolio(df)` accepts a `pandas.DataFrame` with the following columns:

### Identifiers (required)

| Column | Type | Description |
|---|---|---|
| `gebid` | str / int | Unique building identifier |
| `scenario_name` | str | Retrofit scenario name, e.g. `"EGB55 + HP-400-003\|Allgemeiner Strommix"` |
| `scenario_category` | str | `"combined"`, `"construction_only"`, or `"hvac_only"` |
| `enob_category` | str | Target energy efficiency standard, e.g. `"EGB55"` |
| `hvac_system_type` | str / NaN | HVAC system type (NaN for construction_only) |
| `hvac_subsystem_type` | str / NaN | HVAC subsystem type |
| `nfa` | float | Net floor area [m²] (alias: `ngf_sqm`) |
| `bac` | str | Building age class (used for HVAC RSL imputation) |

### Physical criteria (unscaled)

| Column | Unit | Description |
|---|---|---|
| `hvac_rsl` | years | Remaining service life of the existing HVAC system |
| `envbpot` | string or float 0–1 | Envelope improvement potential |
| `endenergy_norm_diff` | kWh/(m²·a) | Specific final energy savings |
| `co2_norm_heating_dhw_diff` | kgCO₂eq/(m²·a) | Specific GHG savings (operational) |
| `penrt_norm_diff` | kWh/(m²·a) | Specific non-renewable primary energy savings |
| `pet_norm_diff` | kWh/(m²·a) | Specific total primary energy savings |
| `pert_norm_diff` | kWh/(m²·a) | Specific renewable primary energy savings |
| `lca_gwp_norm_a2` | kgCO₂eq/m² | Embodied GHG of construction materials |
| `lca_penrt_norm` | kWh/m² | Embodied non-renewable primary energy of materials |
| `lca_pert_norm` | kWh/m² | Embodied renewable primary energy of materials |
| `lcc_norm` | €/m² | Lifecycle cost of materials (per net floor area) |
| `total_pv_potential_kwh` | kWh/a | PV electricity generation potential |
| `investment_costs_total` | € | Total investment cost |
| `investment_costs_norm` | €/m² | Specific investment cost |
| `operation_costs_norm_diff` | €/(m²·a) | Specific operational cost savings |

### Annual totals (for schedule summary)

| Column | Unit | Description |
|---|---|---|
| `endenergy_total_diff` | kWh/a | Total final energy savings |
| `penrt_total_diff` | kWh/a | Total non-renewable primary energy savings |
| `pet_total_diff` | kWh/a | Total primary energy savings |
| `co2_total_heating_dhw_diff` | kgCO₂eq/a | Total GHG savings |
| `operation_costs_total_diff` | €/a | Total operational cost savings |

---

## Output

`apply_weighting_and_order_list` returns a 6-tuple:

| Return value | Description |
|---|---|
| `df_weight` | Input DataFrame with weight and score columns appended |
| `df_retrofit_order` | All building-scenarios ranked by MCDM score |
| `df_schedule` | One-scenario-per-building list with `retrofit_year` column |
| `year_summary` | Per-year aggregates (see table below) |
| `annual_nfa_limit` | Annual retrofit capacity target [m²] |
| `df_retrofit_order_filtered` | `df_retrofit_order` after scenario selection |

### `year_summary` columns

| Column | Description |
|---|---|
| `year` | Retrofit year |
| `buildings_count` | Number of buildings scheduled in that year |
| `total_nfa_m2` | Total floor area retrofitted [m²] |
| `total_nfa_deduplicated_m2` | Floor area with double-counted buildings halved |
| `final_energy_savings_kWh_a` | Cumulative final energy savings [kWh/a] |
| `ghg_savings_kgCO2eq_a` | Cumulative GHG savings [kgCO₂eq/a] |
| `penrt_savings_kWh_a` | Non-renewable primary energy savings [kWh/a] |
| `pet_savings_kWh_a` | Total primary energy savings [kWh/a] |
| `opex_savings_EUR_a` | Operational cost savings [€/a] |
| `investment_costs_EUR` | Total investment required [€] |
| `target_nfa_m2` | Annual capacity target [m²] |
| `actual_nfa_m2` | Actual floor area assigned to this year [m²] |
| `achievement_pct` | Target achievement [%] |

---

## Usage

```python
from retrofit_priority import (
    preprocess_portfolio,
    apply_hierarchical_weighting,
    validate_weighting,
    apply_weighting_and_order_list,
    sum_cluster_points,
    CLUSTER_WEIGHTINGS,
    create_example_dataframe,
)

# 1. Load your portfolio data (or use the built-in example)
df_raw = create_example_dataframe()

# 2. Preprocess: encode categorical fields and scale all criteria
df = preprocess_portfolio(df_raw)

# 3. Choose and validate a weighting scenario
weighting = apply_hierarchical_weighting(CLUSTER_WEIGHTINGS["weighted_stmb"])
validate_weighting(weighting)

# 4. Score, filter, and schedule
df_weight, df_order, df_schedule, year_summary, nfa_limit, df_filtered = \
    apply_weighting_and_order_list(df, weighting, start_year=2026, end_year=2040)

# 5. (Optional) add cluster-level score columns
df_schedule = sum_cluster_points(df_schedule, CLUSTER_WEIGHTINGS["weighted_stmb"])

print(df_order[["gebid", "scenario_name", "points"]].head(10))
print(year_summary)
```

### Custom weighting

```python
from retrofit_priority import create_weighting_dict, validate_weighting

# Direct flat weighting (weights must sum to 1.0)
weighting = create_weighting_dict({
    "hvac_rsl":    0.20,
    "envbpot":     0.20,
    "energy_red":  0.30,
    "gwp_red":     0.30,
})
validate_weighting(weighting)
```

---

## Dependencies

```
python >= 3.10
numpy
pandas
scikit-learn
```

Install via pip:

```bash
pip install numpy pandas scikit-learn
```

---

## Example Data

The file `example_portfolio_50buildings.csv` contains a synthetic portfolio of **50 buildings × 3 scenarios = 150 rows** and can be used to run the algorithm out of the box.

### Portfolio characteristics

| Property | Range / values |
|---|---|
| Buildings | 50 (IDs: B001 – B050) |
| Building types | school, office, sports hall, community center, admin, nursery, library, workshop |
| Building age classes | *vor 1919*, *1919–1948*, *1949–1978*, *1979–1995*, *1996–2009*, *ab 2010* |
| Net floor area (`nfa`) | 400 – 8 000 m² |
| Scenarios per building | 3: `construction_only`, `hvac_only`, `combined` |
| Retrofit measures | EGB55 (envelope), HP-400-003 heat pump (HVAC) |

### Column overview

| Column | Notes |
|---|---|
| `hvac_rsl` | Blank (empty) for `construction_only` rows — load with `pd.to_numeric(..., errors="coerce")` to get `NaN` |
| `endenergy_norm_diff` | Specific final energy savings in kWh/(m²·a); higher for envelope retrofits |
| `investment_costs_total` | Scales with `nfa`; combined scenarios carry a 15 % cost surcharge |
| `lca_gwp_norm_a2` | Embodied GHG of construction materials; non-zero only for scenarios involving an envelope retrofit |

### Loading the example

```python
import pandas as pd
from retrofit_priority import preprocess_portfolio

df_raw = pd.read_csv("example_portfolio_50buildings.csv")
df_raw["hvac_rsl"] = pd.to_numeric(df_raw["hvac_rsl"], errors="coerce")
df = preprocess_portfolio(df_raw)
```

Running `python retrofit_priority.py` from the repository folder automatically loads this CSV.

---

## File Structure

```
retrofit_priority.py                Main module (all functions)
README.md                           This documentation
example_portfolio_50buildings.csv   Synthetic 50-building example portfolio
LICENSE                             MIT License
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

