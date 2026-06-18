"""
retrofit_priority.py
====================
Multi-Criteria Decision Making (MCDM) algorithm for prioritising building
retrofit measures within a portfolio.

Published in:
    Ehlers, N. et al. (2026). "Building Portfolio Retrofit Pathways: 
    A Generic Process Development and Case Study Application."
    Energy and Buildings.

Author:
    Nico Ehlers
    Chair of Energy Efficient and Sustainable Design and Building
    Technical University of Munich
    Arcisstr. 21, 80333 Munich, Germany]
    nico.ehlers@tum.de

License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SCALING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def qt_uniform(series: pd.Series) -> np.ndarray:
    """
    Map a feature to a uniform [0, 1] distribution via the Quantile (Rank)
    Transformer.  Robust against outliers — the most frequent values are spread
    out rather than compressed at the extremes.

    Parameters
    ----------
    series : pd.Series
        1-D input feature (must not contain NaN).

    Returns
    -------
    np.ndarray, shape (n,)
        Uniformly scaled values in [0, 1].
    """
    n = len(series)
    qt = QuantileTransformer(
        n_quantiles=min(1000, n), output_distribution="uniform", random_state=0
    )
    return qt.fit_transform(series.values.reshape(-1, 1)).ravel()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Maps the qualitative building-envelope assessment to a numeric score in [0, 1].
ENVELOPE_POTENTIAL_MAP: dict[str, float] = {
    "Very high":              1.00,
    "High":                   0.75,
    "Medium":               0.50,
    "Medium": 0.50,
    "Low":                0.25,
    "Negligible":                    0.00,
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw building portfolio data for MCDM scoring.

    Performs column aliasing, categorical encoding, NaN imputation for HVAC
    remaining-service-life, and Quantile-Uniform scaling for all criteria.

    Required input columns
    ──────────────────────
    Identifiers:
        gebid                   – building identifier
        scenario_name           – name of the retrofit scenario
        scenario_category       – 'combined' | 'construction_only' | 'hvac_only'
        enob_category           – target energy standard (e.g. 'EGB55')
        hvac_system_type        – HVAC system type (NaN for construction_only)
        hvac_subsystem_type     – HVAC subsystem (NaN for construction_only)
        bac                     – building age class (used for RSL imputation)

    Physical quantities (unscaled):
        nfa                     – net floor area [m²]  (or 'ngf_sqm' as alias)
        hvac_rsl                – remaining HVAC service life [years]
        envbpot                 – envelope improvement potential
                                  (string from ENVELOPE_POTENTIAL_MAP, or float 0-1)
        endenergy_norm_diff     – specific final energy savings [kWh/(m²·a)]
        co2_norm_heating_dhw_diff – specific GHG savings [kgCO₂eq/(m²·a)]
        penrt_norm_diff         – specific non-renewable primary energy savings [kWh/(m²·a)]
        pet_norm_diff           – specific total primary energy savings [kWh/(m²·a)]
        pert_norm_diff          – specific renewable primary energy savings [kWh/(m²·a)]
        lca_gwp_norm_a2         – embodied GHG of materials [kgCO₂eq/m²]
        lca_penrt_norm          – embodied non-renewable primary energy [kWh/m²]
        lca_pert_norm           – embodied renewable primary energy [kWh/m²]
        lcc_norm                – lifecycle cost of materials [€/m²]
        total_pv_potential_kwh  – PV energy potential [kWh/a]
        investment_costs_total  – total investment costs [€]
        investment_costs_norm   – specific investment costs [€/m²]
        operation_costs_norm_diff – specific operational cost savings [€/(m²·a)]

    Totals (used in schedule year-summary only):
        endenergy_total_diff, penrt_total_diff, pet_total_diff,
        co2_total_heating_dhw_diff, operation_costs_total_diff

    Returns
    ───────
    pd.DataFrame
        Copy of the input with all short-name aliases and '_scaled' columns
        appended.
    """
    df = df.copy()

    # Accept 'ngf_sqm' as an alias for net floor area
    if "ngf_sqm" in df.columns and "nfa" not in df.columns:
        df.rename(columns={"ngf_sqm": "nfa"}, inplace=True)

    # Encode qualitative envelope potential → float 0-1; keep the string for reference
    if df["envbpot"].dtype == object:
        df["envbpot_string"] = df["envbpot"].copy()
        df["envbpot"] = df["envbpot"].map(ENVELOPE_POTENTIAL_MAP).astype(float)
    else:
        df["envbpot_string"] = df["envbpot"].astype(str)

    # Measure score: a 'combined' scenario covers both envelope and HVAC → 1.0
    df["measure"] = df["scenario_category"].map(
        {"combined": 1.0, "construction_only": 0.0, "hvac_only": 0.0}
    )

    # Impute missing hvac_rsl values (expected for construction_only) using the
    # mean RSL of buildings in the same age class, then global mean as fallback.
    if df["hvac_rsl"].isna().any():
        bac_means = df.groupby("bac")["hvac_rsl"].mean()
        df["hvac_rsl"] = df["hvac_rsl"].fillna(df["bac"].map(bac_means))
    df["hvac_rsl"] = df["hvac_rsl"].fillna(df["hvac_rsl"].mean()).fillna(0.0)

    # Alias raw performance columns to short names used throughout this module
    aliases = {
        "energy_red":      "endenergy_norm_diff",
        "gwp_red":         "co2_norm_heating_dhw_diff",
        "penrt_red":       "penrt_norm_diff",
        "pet_red":         "pet_norm_diff",
        "pert_red":        "pert_norm_diff",
        "gwp_materials":   "lca_gwp_norm_a2",
        "penrt_materials": "lca_penrt_norm",
        "pert_materials":  "lca_pert_norm",
        "pv_energy":       "total_pv_potential_kwh",
        "invest_cost":     "investment_costs_total",
        "cost_red":        "operation_costs_norm_diff",
        "lcc_materials":   "lcc_norm",
    }
    for short, raw in aliases.items():
        if raw in df.columns:
            df[short] = df[raw]

    # Cost-efficiency: specific energy savings per unit of investment [kWh/(€·m²·a)]
    df["cost_energy_red"] = df["energy_red"] / df["invest_cost"]

    # ── Quantile-Uniform scaling ─────────────────────────────────────────────
    # Higher scaled value always means "more preferable" for that criterion.
    # Criteria where a LOWER physical value is better are inverted (1 – scaled).

    df["hvac_rsl_scaled"]        = 1 - qt_uniform(df["hvac_rsl"])        # short RSL = urgent = high score
    df["energy_red_scaled"]      = qt_uniform(df["energy_red"])
    df["gwp_red_scaled"]         = qt_uniform(df["gwp_red"])
    df["penrt_red_scaled"]       = qt_uniform(df["penrt_red"])
    df["pet_red_scaled"]         = qt_uniform(df["pet_red"])
    df["pert_red_scaled"]        = qt_uniform(df["pert_red"])
    df["gwp_materials_scaled"]   = 1 - qt_uniform(df["gwp_materials"])   # lower embodied GHG = better
    df["penrt_materials_scaled"] = 1 - qt_uniform(df["penrt_materials"])
    df["pert_materials_scaled"]  = 1 - qt_uniform(df["pert_materials"])
    df["pv_energy_scaled"]       = qt_uniform(df["pv_energy"])
    df["invest_cost_scaled"]     = 1 - qt_uniform(df["invest_cost"])     # lower cost = better
    df["cost_energy_red_scaled"] = qt_uniform(df["cost_energy_red"])
    df["cost_red_scaled"]        = qt_uniform(df["cost_red"])
    df["lcc_materials_scaled"]   = 1 - qt_uniform(df["lcc_materials"])   # lower lifecycle cost = better
    df["nfa_scaled"]             = qt_uniform(df["nfa"])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CRITERIA REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

class CriteriaConfig:
    """
    Registry of MCDM criteria: maps each criterion key to its scaled data column,
    paper label (t1–t13 / u5), and a short description.
    """

    def __init__(self):
        # (scaled_column, paper_label, description)
        self._criteria: dict[str, tuple] = {
            "measure":         ("measure",                "u5",  "Retrofit scope – combined scenario = 1.0"),
            "hvac_rsl":        ("hvac_rsl_scaled",        "t1",  "Remaining HVAC service life (inverted: short RSL → high score)"),
            "envbpot":         ("envbpot",                "t2",  "Building-envelope improvement potential"),
            "energy_red":      ("energy_red_scaled",      "t3",  "Specific final energy savings [kWh/(m²·a)]"),
            "penrt_red":       ("penrt_red_scaled",       "t4",  "Specific non-renewable primary energy savings [kWh/(m²·a)]"),
            "gwp_red":         ("gwp_red_scaled",         "t5",  "Specific GHG savings – operational [kgCO₂eq/(m²·a)]"),
            "cost_red":        ("cost_red_scaled",        "t6",  "Specific operational cost savings [€/(m²·a)]"),
            "cost_energy_red": ("cost_energy_red_scaled", "t7",  "Cost efficiency: energy savings per investment [kWh/(€·m²·a)]"),
            "invest_cost":     ("invest_cost_scaled",     "t8",  "Total investment cost (inverted)"),
            "pv_energy":       ("pv_energy_scaled",       "t9",  "PV energy potential [kWh/a]"),
            "penrt_materials": ("penrt_materials_scaled", "t10", "Embodied non-renewable primary energy (inverted)"),
            "gwp_materials":   ("gwp_materials_scaled",   "t11", "Embodied GHG of materials (inverted)"),
            "lcc_materials":   ("lcc_materials_scaled",   "t12", "Lifecycle cost of materials (inverted)"),
            "pet_red":         ("pet_red_scaled",         "t13", "Specific total primary energy savings [kWh/(m²·a)]"),
        }

    def get_data_column(self, key: str) -> str:
        return self._criteria[key][0]

    def available_criteria(self) -> list[str]:
        return list(self._criteria.keys())

    def display_name(self, key: str) -> str:
        return self._criteria[key][1]


criteria_config = CriteriaConfig()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  WEIGHTING & SCALARISATION
# ──────────────────────────────────────────────────────────────────────────────

def create_weighting_dict(criteria_weights: dict) -> dict:
    """
    Prefix criterion keys with 'f_' to produce a weighting dictionary.

    Parameters
    ----------
    criteria_weights : dict
        {criterion_key: weight}, e.g. {'hvac_rsl': 0.3, 'energy_red': 0.7}

    Returns
    -------
    dict
        {'f_hvac_rsl': 0.3, 'f_energy_red': 0.7}
    """
    return {f"f_{k}": v for k, v in criteria_weights.items()}


def validate_weighting(weighting_dict: dict) -> bool:
    """
    Check that all weights sum to 1.0 and every key refers to a known criterion.

    Raises ``ValueError`` on failure; returns ``True`` when valid.
    """
    total = sum(weighting_dict.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
    known = {f"f_{k}" for k in criteria_config.available_criteria()}
    unknown = set(weighting_dict) - known
    if unknown:
        raise ValueError(f"Unknown criteria in weighting dict: {unknown}")
    return True


def apply_hierarchical_weighting(cluster_weighting: dict) -> dict:
    """
    Flatten a two-level (cluster → criteria) weighting into a single flat
    weighting dictionary (the scalarisation step).

    Each criterion's final weight is::

        w_final = cluster_weight × w_within_cluster

    Criteria that appear in multiple clusters have their contributions summed.

    Parameters
    ----------
    cluster_weighting : dict
        Nested definition, for example::

            {
                'Priority': {
                    'cluster_weight': 0.30,
                    'criteria_weights': {'hvac_rsl': 0.30, 'envbpot': 0.50, 'measure': 0.20}
                },
                'Performance': {
                    'cluster_weight': 0.70,
                    'criteria_weights': {'energy_red': 0.50, 'gwp_red': 0.50}
                }
            }

        - Cluster weights must sum to 1.0.
        - Criteria weights within each cluster must sum to 1.0.

    Returns
    -------
    dict
        Flat weighting dict, e.g. {'f_hvac_rsl': 0.09, 'f_energy_red': 0.35, ...}
    """
    total_cluster = sum(c["cluster_weight"] for c in cluster_weighting.values())
    if not (0.99 <= total_cluster <= 1.01):
        raise ValueError(f"Cluster weights must sum to 1.0, got {total_cluster:.4f}")

    flat: dict[str, float] = {}
    for cluster in cluster_weighting.values():
        cw = cluster["cluster_weight"]
        inner = cluster["criteria_weights"]
        total_inner = sum(inner.values())
        if not (0.99 <= total_inner <= 1.01):
            raise ValueError(
                f"Criteria weights within a cluster must sum to 1.0, got {total_inner:.4f}"
            )
        for criterion, weight in inner.items():
            flat[criterion] = flat.get(criterion, 0.0) + weight * cw

    return create_weighting_dict(flat)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  SCENARIO SELECTION  (one scenario per building)
# ──────────────────────────────────────────────────────────────────────────────

def filter_scenarios_by_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce each building to exactly one retrofit scenario.

    The input ``df`` must be sorted by 'points' in **descending** order before
    calling this function (highest score first).

    Selection rules
    ───────────────
    1. **combined on top** → keep the highest-ranked combined scenario.
    2. **construction_only on top** → look for the combined scenario that pairs
       the best construction and best hvac_only measures.  If found, copy the
       combined's performance metrics (energy/GHG/cost reductions) into the
       hvac_only row and keep that row (so the schedule benefits from the HVAC
       replacement cost data).  Falls back to the construction_only row if no
       matching combined+hvac pair exists.
    3. **hvac_only on top** → prefer the combined scenario whose name contains
       the top hvac_only name; fall back to the highest combined, or the
       hvac_only row if no combined exists.

    Parameters
    ----------
    df : pd.DataFrame
        Scored building-scenario table with columns:
        'gebid', 'scenario_category', 'scenario_name',
        'hvac_system_type', 'hvac_subsystem_type'.

    Returns
    -------
    pd.DataFrame
        One row per building, preserving the original sort order.
    """
    df_out = df.copy()
    building_top = df_out.groupby("gebid", sort=False).first()[
        ["scenario_category", "scenario_name"]
    ]

    rows_to_keep: list = []

    for gebid in df_out["gebid"].unique():
        building_rows = df_out[df_out["gebid"] == gebid]
        top_category  = building_top.loc[gebid, "scenario_category"]
        top_name      = building_top.loc[gebid, "scenario_name"]

        combined_idxs = building_rows[building_rows["scenario_category"] == "combined"].index.tolist()
        cons_idxs     = building_rows[building_rows["scenario_category"] == "construction_only"].index.tolist()
        hvac_idxs     = building_rows[building_rows["scenario_category"] == "hvac_only"].index.tolist()

        keep_idx = None

        if top_category == "combined":
            keep_idx = combined_idxs[0] if combined_idxs else None

        elif top_category == "construction_only":
            hvac_idx = hvac_idxs[0] if hvac_idxs else None
            matched_combined_idx = None

            if hvac_idx is not None and combined_idxs:
                cons_row = df_out.loc[cons_idxs[0]] if cons_idxs else pd.Series(dtype=object)
                hvac_row = df_out.loc[hvac_idx]

                # Only compare columns that are present in the DataFrame
                compare_cols = [
                    c for c in ["retrofit_measures", "hvac_system_type", "hvac_subsystem_type"]
                    if c in df_out.columns
                ]
                cons_sn = str(cons_row.get("scenario_name", "")) if not cons_row.empty else ""
                hvac_sn = str(hvac_row.get("scenario_name", ""))

                for cidx in combined_idxs:
                    comb_row = df_out.loc[cidx]
                    comb_sn  = str(comb_row.get("scenario_name", ""))
                    match = True

                    # The combined scenario name must contain both constituent names
                    if cons_sn and hvac_sn and comb_sn:
                        if cons_sn not in comb_sn or hvac_sn not in comb_sn:
                            match = False

                    # Each combined value for structural columns must originate
                    # from either the construction_only or hvac_only row
                    if match:
                        for col in compare_cols:
                            v_comb = comb_row.get(col, np.nan)
                            v_cons = cons_row.get(col, np.nan) if not cons_row.empty else np.nan
                            v_hvac = hvac_row.get(col, np.nan)
                            if pd.isna(v_comb) and pd.isna(v_cons) and pd.isna(v_hvac):
                                continue
                            if not (
                                (not pd.isna(v_cons) and v_comb == v_cons)
                                or (not pd.isna(v_hvac) and v_comb == v_hvac)
                            ):
                                match = False
                                break

                    if match:
                        matched_combined_idx = cidx
                        break

            if matched_combined_idx is not None and hvac_idx is not None:
                # Copy the combined scenario's performance savings into the hvac_only row,
                # then keep that row (it already contains HVAC investment cost data).
                perf_cols = [
                    c for c in
                    ["penrt_red", "energy_red", "gwp_red", "cost_red", "cost_energy_red"]
                    if c in df_out.columns
                ]
                for col in perf_cols:
                    df_out.at[hvac_idx, col] = df_out.at[matched_combined_idx, col]
                keep_idx = hvac_idx
            elif cons_idxs:
                keep_idx = cons_idxs[0]

        elif top_category == "hvac_only":
            hvac_sn = str(top_name) if pd.notna(top_name) else ""
            if combined_idxs:
                keep_idx = next(
                    (cidx for cidx in combined_idxs
                     if hvac_sn and hvac_sn in str(df_out.at[cidx, "scenario_name"])),
                    combined_idxs[0],  # fallback to highest-ranked combined
                )
            elif hvac_idxs:
                keep_idx = hvac_idxs[0]

        else:
            keep_idx = building_rows.index[0]

        if keep_idx is not None:
            rows_to_keep.append(keep_idx)

    keep_set = set(rows_to_keep)
    ordered  = [i for i in df_out.index if i in keep_set]
    return df_out.loc[ordered].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  RETROFIT SCHEDULING  (greedy bin-packing with look-ahead)
# ──────────────────────────────────────────────────────────────────────────────

def retrofit_schedule(
    df: pd.DataFrame,
    start_year: int = 2026,
    end_year: int | None = None,
    quotient: float = 0.03,
    threshold_percent: float = 0.0,
) -> tuple:
    """
    Assign a retrofit year to every building using a greedy bin-packing algorithm.

    Buildings are processed in descending MCDM-score order.  Each year's
    capacity target is filled greedily; when a building would exceed the target,
    a look-ahead pass scans all remaining buildings for any that still fit within
    the allowed threshold before closing the year.

    Capacity modes
    ──────────────
    - **Quotient mode** (``end_year=None``):
      ``annual_capacity = total_nfa × quotient``
    - **End-year mode** (``end_year`` is set):
      ``annual_capacity = total_nfa / (end_year − start_year + 1)``

    Parameters
    ----------
    df : pd.DataFrame
        Ranked building list (descending by 'points') containing at minimum
        'nfa', 'gebid', and 'points'.
    start_year : int
    end_year : int or None
    quotient : float
        Annual retrofit rate as fraction of total portfolio floor area.
    threshold_percent : float
        Allowed over- or under-shoot of the annual capacity target [%].

    Returns
    -------
    annual_nfa_limit : float
    df_schedule : pd.DataFrame
        Input DataFrame with 'retrofit_year' column added.
    year_summary : pd.DataFrame
        Per-year aggregates (counts, floor area, energy/GHG/cost savings).
    last_year : int
    """
    total_nfa = df["nfa"].sum()
    if end_year is None:
        annual_nfa_limit = total_nfa * quotient
    else:
        annual_nfa_limit = total_nfa / (end_year - start_year + 1)

    df_schedule  = df.copy().reset_index(drop=True)
    df_schedule["retrofit_year"] = None
    df_remaining = df_schedule.copy()

    current_year = start_year
    year_records: list[dict] = []

    while not df_remaining.empty:
        max_allowed   = annual_nfa_limit * (1 + threshold_percent / 100)
        selected: list = []
        cumulative    = 0.0

        for idx, row in df_remaining.iterrows():
            building_nfa  = row["nfa"]
            new_total     = cumulative + building_nfa

            if new_total <= max_allowed or len(selected) == 0:
                selected.append(idx)
                cumulative = new_total
                # Stop once we have reached a sufficient fill level
                if cumulative >= annual_nfa_limit * (1 - threshold_percent / 100):
                    break
                # If the very first building already overshoots, stop
                if len(selected) == 1 and cumulative > max_allowed:
                    break
            else:
                # This building doesn't fit; scan all remaining buildings for
                # any that still fit within the allowed capacity.
                for offset in range(1, len(df_remaining)):
                    next_idx = idx + offset
                    if next_idx in df_remaining.index:
                        next_nfa = df_remaining.loc[next_idx, "nfa"]
                        if cumulative + next_nfa <= max_allowed:
                            selected.append(next_idx)
                            cumulative += next_nfa
                break  # close year after look-ahead pass

        # Safety: always advance by at least one building to avoid an infinite loop
        if not selected:
            selected   = [df_remaining.index[0]]
            cumulative = df_remaining.iloc[0]["nfa"]

        actual_nfa = df_remaining.loc[
            pd.Index(selected).intersection(df_remaining.index), "nfa"
        ].sum()

        df_schedule.loc[
            pd.Index(selected).intersection(df_schedule.index), "retrofit_year"
        ] = current_year
        df_remaining = df_remaining.drop(
            pd.Index(selected).intersection(df_remaining.index)
        )

        year_records.append(
            {
                "retrofit_year":   current_year,
                "target_nfa_m2":   annual_nfa_limit,
                "actual_nfa_m2":   actual_nfa,
                "achievement_pct": (actual_nfa / annual_nfa_limit * 100)
                                    if annual_nfa_limit > 0 else 0.0,
            }
        )
        current_year += 1
        if current_year > start_year + 200:  # safety break
            break

    # Buildings with the same gebid (e.g., HVAC + construction as separate rows)
    # contribute halved floor area to avoid double-counting in the portfolio total.
    df_schedule["nfa_counted"] = df_schedule["nfa"]
    duplicated = df_schedule["gebid"].duplicated(keep=False)
    df_schedule.loc[duplicated, "nfa_counted"] = df_schedule.loc[duplicated, "nfa"] / 2

    # ── Year summary ──────────────────────────────────────────────────────────
    agg_map = {
        "nfa":                          "sum",
        "nfa_counted":                  "sum",
        "gebid":                        "count",
        "co2_total_heating_dhw_diff":   "sum",
        "endenergy_total_diff":         "sum",
        "penrt_total_diff":             "sum",
        "pet_total_diff":               "sum",
        "operation_costs_total_diff":   "sum",
        "investment_costs_total":       "sum",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in df_schedule.columns}

    year_summary = (
        df_schedule.groupby("retrofit_year")
        .agg(agg_map)
        .reset_index()
        .merge(pd.DataFrame(year_records), on="retrofit_year", how="left")
    )

    # Human-readable column rename
    year_summary.rename(
        columns={
            "retrofit_year":                "year",
            "nfa":                          "total_nfa_m2",
            "nfa_counted":                  "total_nfa_deduplicated_m2",
            "gebid":                        "buildings_count",
            "co2_total_heating_dhw_diff":   "ghg_savings_kgCO2eq_a",
            "endenergy_total_diff":         "final_energy_savings_kWh_a",
            "penrt_total_diff":             "penrt_savings_kWh_a",
            "pet_total_diff":               "pet_savings_kWh_a",
            "operation_costs_total_diff":   "opex_savings_EUR_a",
            "investment_costs_total":       "investment_costs_EUR",
        },
        inplace=True,
    )

    df_schedule = df_schedule.sort_values("points", ascending=False)
    return annual_nfa_limit, df_schedule, year_summary, current_year


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN MCDM FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def apply_weighting_and_order_list(
    df_weight: pd.DataFrame,
    weighting: dict,
    start_year: int = 2026,
    end_year: int | None = None,
    quotient: float = 0.03,
) -> tuple:
    """
    Score buildings by weighted linear aggregation, filter to one scenario per
    building, and produce a retrofit schedule.

    For each criterion *key* in *weighting* (prefixed 'f_'), the score is::

        <criterion>_points = f_<criterion> × <criterion>_scaled

    The total MCDM score is the sum of all criterion-specific points::

        points = Σ <criterion>_points

    Buildings with a higher total score are scheduled earlier.

    Notes
    ─────
    *  ``construction_only`` scenarios have no HVAC replacement, so their
       HVAC RSL points are forced to zero regardless of the weight.
    *  ``df_weight`` is modified **in place** (weight and score columns are
       appended); pass a copy if you need to preserve the original.

    Parameters
    ----------
    df_weight : pd.DataFrame
        Preprocessed portfolio DataFrame (output of ``preprocess_portfolio``).
    weighting : dict
        Flat weighting dict {'f_<criterion>': weight, ...} summing to 1.0.
        Produced by ``apply_hierarchical_weighting`` or ``create_weighting_dict``.
    start_year : int
    end_year : int or None
    quotient : float

    Returns
    ───────
    df_weight : pd.DataFrame
        Input DataFrame with weight and score columns added.
    df_retrofit_order : pd.DataFrame
        All scenarios ranked by total MCDM score.
    df_schedule : pd.DataFrame
        One-scenario-per-building list with 'retrofit_year' column.
    year_summary : pd.DataFrame
        Per-year aggregate statistics.
    annual_nfa_limit : float
        Annual retrofit capacity [m²].
    df_retrofit_order_filtered : pd.DataFrame
        ``df_retrofit_order`` after scenario-selection rules are applied.
    """
    # Initialise criterion point columns to zero
    point_columns: list[str] = []
    for weight_key in weighting:
        point_col = weight_key.replace("f_", "") + "_points"
        point_columns.append(point_col)
        df_weight[point_col] = 0.0

    # Weighted score per criterion
    for weight_key, weight_value in weighting.items():
        df_weight[weight_key] = weight_value          # stored for traceability
        point_col = weight_key.replace("f_", "") + "_points"

        # envbpot and measure are already in [0, 1]; others use the '_scaled' column
        if weight_key == "f_envbpot":
            data_col = "envbpot"
        elif weight_key == "f_measure":
            data_col = "measure"
        else:
            data_col = weight_key.replace("f_", "") + "_scaled"

        if data_col in df_weight.columns:
            df_weight[point_col] = df_weight[data_col] * weight_value

    # construction_only has no HVAC replacement → zero HVAC-RSL points
    if df_weight["scenario_category"].eq("construction_only").any():
        if "hvac_rsl_points" in df_weight.columns:
            df_weight.loc[df_weight["scenario_category"] == "construction_only", "hvac_rsl_points"] = 0.0

    df_weight["points"] = df_weight[point_columns].sum(axis=1)

    # Select output columns: identifiers + raw values + weights + scores
    base_cols = [
        "gebid", "scenario_name", "scenario_category", "enob_category",
        "hvac_system_type", "hvac_subsystem_type", "nfa",
        "bac", "hvac_rsl", "hvac_rsl_scaled",
        "measure", "envbpot_string", "envbpot",
        "invest_cost", "pv_energy",
        "penrt_materials", "gwp_materials", "lcc_materials",
        "energy_red", "penrt_red", "pet_red", "gwp_red", "cost_red", "cost_energy_red",
        "endenergy_total_diff", "penrt_total_diff", "pet_total_diff",
        "co2_total_heating_dhw_diff", "operation_costs_total_diff",
        "investment_costs_total",
    ]
    base_cols  = [c for c in base_cols if c in df_weight.columns]
    extra_cols = list(weighting.keys()) + point_columns + ["points"]

    df_retrofit_order = df_weight[base_cols + extra_cols].copy()
    df_retrofit_order.sort_values("points", ascending=False, inplace=True)
    df_retrofit_order.reset_index(drop=True, inplace=True)

    # Reduce to one scenario per building
    df_retrofit_order_filtered = filter_scenarios_by_rules(df_retrofit_order)
    df_retrofit_order_filtered.sort_values("points", ascending=False, inplace=True)

    annual_nfa_limit, df_schedule, year_summary, _ = retrofit_schedule(
        df_retrofit_order_filtered,
        start_year=start_year,
        end_year=end_year,
        quotient=quotient,
    )

    return (
        df_weight,
        df_retrofit_order,
        df_schedule,
        year_summary,
        annual_nfa_limit,
        df_retrofit_order_filtered,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9.  CLUSTER SCORE AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def sum_cluster_points(
    df: pd.DataFrame,
    cluster_def: dict,
    suffix: str = "_points",
) -> pd.DataFrame:
    """
    Sum individual criterion scores into thematic cluster totals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing criterion point columns (e.g. 'energy_red_points').
    cluster_def : dict
        One entry from ``CLUSTER_WEIGHTINGS``.
    suffix : str
        Suffix of the criterion score columns.

    Returns
    ───────
    pd.DataFrame
        Input DataFrame with a '<cluster_name>_cluster_points' column added
        for each cluster in *cluster_def*.
    """
    result = df.copy()
    for cluster_name, cluster_info in cluster_def.items():
        cols = [
            f"{c}{suffix}"
            for c in cluster_info["criteria_weights"]
            if f"{c}{suffix}" in df.columns
        ]
        result[f"{cluster_name}_cluster_points"] = result[cols].sum(axis=1) if cols else 0.0
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 10.  PREDEFINED WEIGHTING SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

CLUSTER_WEIGHTINGS: dict = {
    # Differentiated: emphasises performance (40 %) and priority criteria (30 %)
    "weighted_stmb": {
        "Priority":      {"cluster_weight": 0.30, "criteria_weights": {"hvac_rsl": 0.30, "envbpot": 0.50, "measure": 0.20}},
        "NFA dependent": {"cluster_weight": 0.20, "criteria_weights": {"invest_cost": 0.60, "pv_energy": 0.40}},
        "Materials":     {"cluster_weight": 0.10, "criteria_weights": {"lcc_materials": 0.70, "gwp_materials": 0.30}},
        "Performance":   {"cluster_weight": 0.40, "criteria_weights": {"energy_red": 0.40, "gwp_red": 0.30, "cost_energy_red": 0.30}},
    },
    # Balanced: equal cluster weights
    "balanced": {
        "Priority":      {"cluster_weight": 0.25, "criteria_weights": {"hvac_rsl": 1 / 3, "envbpot": 1 / 3, "measure": 1 / 3}},
        "NFA dependent": {"cluster_weight": 0.25, "criteria_weights": {"invest_cost": 0.50, "pv_energy": 0.50}},
        "Materials":     {"cluster_weight": 0.25, "criteria_weights": {"gwp_materials": 0.50, "lcc_materials": 0.50}},
        "Performance":   {"cluster_weight": 0.25, "criteria_weights": {"energy_red": 1 / 3, "gwp_red": 1 / 3, "cost_energy_red": 1 / 3}},
    },
    # Extreme: performance-dominated (60 %)
    "extreme": {
        "Priority":      {"cluster_weight": 0.30, "criteria_weights": {"hvac_rsl": 0.30, "envbpot": 0.70, "measure": 0.00}},
        "NFA dependent": {"cluster_weight": 0.10, "criteria_weights": {"invest_cost": 0.80, "pv_energy": 0.20}},
        "Materials":     {"cluster_weight": 0.00, "criteria_weights": {"lcc_materials": 0.90, "gwp_materials": 0.10}},
        "Performance":   {"cluster_weight": 0.60, "criteria_weights": {"energy_red": 0.60, "gwp_red": 0.20, "cost_energy_red": 0.20}},
    },
    # Simple: HVAC urgency focus only
    "simple_rsl": {
        " ": {"cluster_weight": 1.0, "criteria_weights": {"hvac_rsl": 0.70, "envbpot": 0.10, "energy_red": 0.10, "gwp_red": 0.10}},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 11.  EXAMPLE DATA & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def create_example_dataframe() -> pd.DataFrame:
    """
    Build a minimal synthetic building portfolio for testing.

    Generates 5 buildings × 3 scenario types (construction_only, hvac_only,
    combined) with plausible but artificial values for all required input columns.

    Returns
    ───────
    pd.DataFrame  (15 rows × required input columns)
    """
    rng = np.random.default_rng(42)
    buildings = [f"B{i:03d}" for i in range(1, 6)]
    scenario_types = ["construction_only", "hvac_only", "combined"]
    scenario_names = {
        "construction_only": "EGB55",
        "hvac_only":         "HP-400-003|Allgemeiner Strommix",
        "combined":          "EGB55 + HP-400-003|Allgemeiner Strommix",
    }

    rows: list[dict] = []
    for bid in buildings:
        nfa     = float(rng.uniform(500, 5000))
        hvac_rsl = float(rng.uniform(0, 25))
        bac     = rng.choice(["vor 1919", "1919–1948", "1949–1978", "1979–1995"])

        for sc in scenario_types:
            is_hvac = sc in ("hvac_only", "combined")
            is_cons = sc in ("construction_only", "combined")

            energy_red = float(rng.uniform(20, 120) if is_cons else rng.uniform(5, 40))
            invest     = nfa * float(rng.uniform(150, 600))

            rows.append(
                {
                    # ── identifiers ──────────────────────────────────────────
                    "gebid":                       bid,
                    "scenario_name":               scenario_names[sc],
                    "scenario_category":           sc,
                    "enob_category":               rng.choice(["EGB55", "EGB40", "EGB30"]),
                    "hvac_system_type":            "Wärmepumpe" if is_hvac else None,
                    "hvac_subsystem_type":         "Luft"       if is_hvac else None,
                    "nfa":                         nfa,
                    "bac":                         bac,
                    # ── HVAC remaining service life ──────────────────────────
                    # NaN for construction_only — will be imputed in preprocess_portfolio
                    "hvac_rsl":                    hvac_rsl if is_hvac else np.nan,
                    # ── envelope potential (qualitative) ─────────────────────
                    "envbpot":                     rng.choice(list(ENVELOPE_POTENTIAL_MAP.keys())),
                    # ── specific performance indicators ──────────────────────
                    "endenergy_norm_diff":         energy_red,
                    "co2_norm_heating_dhw_diff":   energy_red * float(rng.uniform(0.20, 0.40)),
                    "penrt_norm_diff":             energy_red * float(rng.uniform(2.0, 3.0)),
                    "pet_norm_diff":               energy_red * float(rng.uniform(2.5, 3.5)),
                    "pert_norm_diff":              energy_red * float(rng.uniform(0.1, 0.3)),
                    # ── embodied impacts of construction materials ────────────
                    "lca_gwp_norm_a2":             float(rng.uniform(10,  80)),
                    "lca_penrt_norm":              float(rng.uniform(100, 800)),
                    "lca_pert_norm":               float(rng.uniform(10,  80)),
                    "lcc_norm":                    float(rng.uniform(50,  400)),
                    # ── PV potential & costs ──────────────────────────────────
                    "total_pv_potential_kwh":      nfa * float(rng.uniform(50, 200)),
                    "investment_costs_total":      invest,
                    "investment_costs_norm":       invest / nfa,
                    "operation_costs_norm_diff":   float(rng.uniform(1, 15)),
                    # ── absolute totals for year_summary ─────────────────────
                    "endenergy_total_diff":        energy_red * nfa,
                    "penrt_total_diff":            energy_red * nfa * float(rng.uniform(2.0, 3.0)),
                    "pet_total_diff":              energy_red * nfa * float(rng.uniform(2.5, 3.5)),
                    "co2_total_heating_dhw_diff":  energy_red * nfa * float(rng.uniform(0.2, 0.4)),
                    "operation_costs_total_diff":  float(rng.uniform(1, 15)) * nfa,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os

    # ── 1. Load portfolio data ────────────────────────────────────────────────
    # Looks for the bundled example CSV next to this script; falls back to
    # the programmatic generator if the file is not found.
    _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "example_portfolio_50buildings.csv")
    if os.path.exists(_csv):
        df_raw = pd.read_csv(_csv)
        # hvac_rsl is blank for construction_only rows in the CSV → convert to NaN
        df_raw["hvac_rsl"] = pd.to_numeric(df_raw["hvac_rsl"], errors="coerce")
        print(f"Loaded: {_csv}")
    else:
        print("example_portfolio_50buildings.csv not found – generating synthetic data.")
        df_raw = create_example_dataframe()

    print(f"Portfolio: {df_raw['gebid'].nunique()} buildings, {len(df_raw)} scenarios")

    # ── 2. Preprocess (encode + scale) ───────────────────────────────────────
    df_preprocessed = preprocess_portfolio(df_raw)

    # ── 3. Define weighting scenario ─────────────────────────────────────────
    weighting = apply_hierarchical_weighting(CLUSTER_WEIGHTINGS["weighted_stmb"])
    validate_weighting(weighting)
    print(f"\nWeighting scenario 'weighted_stmb':")
    for k, v in weighting.items():
        label = criteria_config.display_name(k.replace("f_", ""))
        print(f"  {label} ({k}): {v:.4f}")

    # ── 4. Score, filter and schedule ────────────────────────────────────────
    df_weight, df_order, df_schedule, year_summary, nfa_limit, df_filtered = \
        apply_weighting_and_order_list(
            df_preprocessed,
            weighting,
            start_year=2026,
            end_year=2040,
        )

    # ── 5. Aggregate cluster scores ───────────────────────────────────────────
    df_schedule = sum_cluster_points(df_schedule, CLUSTER_WEIGHTINGS["weighted_stmb"])

    # ── 6. Print results ──────────────────────────────────────────────────────
    print(f"\nAnnual retrofit capacity: {nfa_limit:,.0f} m²")
    print(f"\nRetrofit order – all scenarios (top 10 of {len(df_order)}):")
    print(df_order[["gebid", "scenario_name", "points"]].head(10).to_string(index=False))
    print(f"\nFiltered (one scenario per building, {len(df_filtered)} buildings):")
    print(df_filtered[["gebid", "scenario_name", "points"]].to_string(index=False))
    print("\nYear summary:")
    print(year_summary.to_string(index=False))
