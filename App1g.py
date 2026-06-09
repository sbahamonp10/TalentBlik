from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "denmark_player_pool_union.parquet"
WYSCOUT_CATALOG_PATH = DATA_DIR / "wyscout_metrics_categories_updated.xlsx"
APP_TITLE = "TalentBlik"
ROLE_TEMPLATES_PATH = DATA_DIR / "player_role_templates_v30.xlsx"

# IMPORTANT v11
# Same raw-reference scoring logic as v10. Do NOT use percentiles.
# Clear/restart the app after changing this file so old role templates/radars/scores are not cached.
#
# Role-fit and radar must use:
# - TOTAL_SEASON_DF / Wyscout only
# - metric_name for dataframe lookup
# - reference_low / reference_high
# - direction
# - weight
# - RADAR_REFERENCE for radar
#
# Scoring:
# score01 = (raw_value - reference_low) / (reference_high - reference_low)
# score01 = max(0, min(1, score01))
# metric_score = score01 * 100
# if direction == lower: metric_score = 100 - metric_score
# role_score = sum(metric_score * weight) / sum(available_weights)
#
# DMF role_ids now use:
# - dmf_shielder_non_progressor
# - dmf_recoverer_non_progressor
# Visible role names:
# - Shielder + non progressor
# - Recoverer + non progressor
#
# Main Role Signals should show raw metric values only, not internal S scores.

# ============================================================
# DBU COLOR SYSTEM
# ============================================================
DBU_RED = "#d40000"
DBU_RED_DARK = "#8b0000"
DBU_RED_SOFT = "#ff4d4d"
DBU_BG = "#07090d"
DBU_PANEL = "#12151c"
DBU_PANEL_2 = "#171b24"
DBU_BORDER = "#2b2f3a"
DBU_TEXT = "#f8fafc"
DBU_MUTED = "#a1a1aa"
DBU_PITCH_TOP = "#7f0000"
DBU_PITCH_MID = "#3b0a0e"
DBU_PITCH_BOTTOM = "#14080a"

SOURCE_LABELS = {"league": "League only", "total": "Total season"}

# ============================================================
# POSITION SYSTEM
# ============================================================
POSITION_DISPLAY_MAP = {
    "RWF": "RW",
    "LWF": "LW",
    "CM": "CMF",
    "AM": "AMF",
    "DM": "DMF",
}

PITCH_FILTER_MAP = {
    "RW": ["RW", "RWF"],
    "LW": ["LW", "LWF"],
    "CMF": ["CM", "CMF"],
    "AMF": ["AM", "AMF"],
    "DMF": ["DM", "DMF"],
}

PITCH_ROWS = [
    ["CF"],
    ["LW", "LAMF", "AMF", "RAMF", "RW"],
    ["LMF", "LCMF", "CMF", "RCMF", "RMF"],
    ["LWB", "LDMF", "DMF", "RDMF", "RWB"],
    ["LB", "LCB", "CB", "RCB", "RB"],
    ["GK"],
]

# ============================================================
# LEAGUE ONLY / FOTMOB CATALOG HELPERS
# ============================================================
def fm(value=None, per90=None, percentile=None, percentile_per90=None, is_percentage=False):
    out = {}
    if value:
        out["value"] = value
    if per90:
        out["per90"] = per90
    if percentile:
        out["percentile"] = percentile
    if percentile_per90:
        out["percentile_per90"] = percentile_per90
    if is_percentage:
        out["is_percentage"] = True
    return out


LEAGUE_METRIC_CATALOG_RAW = {
    "Shooting": {
        "Goals": fm("fotmob_goals_league", "fotmob_goals_per90_league", "fotmob_goals_pct_league", "fotmob_goals_pct90_league"),
        "Expected goals (xG)": fm("fotmob_xg_league", "fotmob_expected_goals_per90_league", "fotmob_expected_goals_pct_league", "fotmob_expected_goals_pct90_league"),
        "xG on target (xGOT)": fm("fotmob_expected_goals_on_target_league", "fotmob_expected_goals_on_target_per90_league", "fotmob_expected_goals_on_target_pct_league", "fotmob_expected_goals_on_target_pct90_league"),
        "Non-penalty xG": fm("fotmob_npxg_league", "fotmob_non_penalty_xg_per90_league", "fotmob_non_penalty_xg_pct_league", "fotmob_non_penalty_xg_pct90_league"),
        "Shots": fm("fotmob_shots_league", "fotmob_shots_per90_league", "fotmob_shots_pct_league", "fotmob_shots_pct90_league"),
        "Shots on target": fm("fotmob_ShotsOnTarget_league", "fotmob_ShotsOnTarget_per90_league", "fotmob_ShotsOnTarget_pct_league", "fotmob_ShotsOnTarget_pct90_league"),
        "Headed shots": fm("fotmob_headed_shots_league", "fotmob_headed_shots_per90_league", "fotmob_headed_shots_pct_league", "fotmob_headed_shots_pct90_league"),
    },
    "Passing": {
        "Assists": fm("fotmob_assists_league", "fotmob_assists_per90_league", "fotmob_assists_pct_league", "fotmob_assists_pct90_league"),
        "Expected assists (xA)": fm("fotmob_xa_league", "fotmob_expected_assists_per90_league", "fotmob_expected_assists_pct_league", "fotmob_expected_assists_pct90_league"),
        "Successful passes": fm("fotmob_successful_passes_league", "fotmob_successful_passes_per90_league", "fotmob_successful_passes_pct_league", "fotmob_successful_passes_pct90_league"),
        "Successful passes %": fm("fotmob_successful_passes_accuracy_league", "fotmob_successful_passes_accuracy_per90_league", "fotmob_successful_passes_accuracy_pct_league", "fotmob_successful_passes_accuracy_pct90_league", True),
        "Accurate long balls": fm("fotmob_long_balls_accurate_league", "fotmob_long_balls_accurate_per90_league", "fotmob_long_balls_accurate_pct_league", "fotmob_long_balls_accurate_pct90_league"),
        "Accurate long balls %": fm("fotmob_long_ball_succeeeded_accuracy_league", "fotmob_long_ball_succeeeded_accuracy_per90_league", "fotmob_long_ball_succeeeded_accuracy_pct_league", "fotmob_long_ball_succeeeded_accuracy_pct90_league", True),
        "Chances created": fm("fotmob_chances_created_league", "fotmob_chances_created_per90_league", "fotmob_chances_created_pct_league", "fotmob_chances_created_pct90_league"),
        "Big chances created": fm("fotmob_big_chances_created_league", "fotmob_big_chances_created_per90_league", "fotmob_big_chances_created_pct_league", "fotmob_big_chances_created_pct90_league"),
    },
    "Possession": {
        "Successful dribbles": fm("fotmob_dribbles_succeeded_league", "fotmob_dribbles_succeeded_per90_league", "fotmob_dribbles_succeeded_pct_league", "fotmob_dribbles_succeeded_pct90_league"),
        "Successful dribbles %": fm("fotmob_dribbles_succeeded_percent_league", "fotmob_dribbles_succeeded_percent_per90_league", "fotmob_dribbles_succeeded_percent_pct_league", "fotmob_dribbles_succeeded_percent_pct90_league", True),
        "Duels won": fm("fotmob_duel_won_league", "fotmob_duel_won_per90_league", "fotmob_duel_won_pct_league", "fotmob_duel_won_pct90_league"),
        "Duels won %": fm("fotmob_duel_won_percent_league", "fotmob_duel_won_percent_per90_league", "fotmob_duel_won_percent_pct_league", "fotmob_duel_won_percent_pct90_league", True),
        "Aerial duels won": fm("fotmob_aerials_won_league", "fotmob_aerials_won_per90_league", "fotmob_aerials_won_pct_league", "fotmob_aerials_won_pct90_league"),
        "Aerial duels won %": fm("fotmob_aerials_won_percent_league", "fotmob_aerials_won_percent_per90_league", "fotmob_aerials_won_percent_pct_league", "fotmob_aerials_won_percent_pct90_league", True),
        "Touches": fm("fotmob_touches_league", "fotmob_touches_per90_league", "fotmob_touches_pct_league", "fotmob_touches_pct90_league"),
        "Touches in opposition box": fm("fotmob_touches_opp_box_league", "fotmob_touches_opp_box_per90_league", "fotmob_touches_opp_box_pct_league", "fotmob_touches_opp_box_pct90_league"),
        "Dispossessed": fm("fotmob_dispossessed_league", "fotmob_dispossessed_per90_league", "fotmob_dispossessed_pct_league", "fotmob_dispossessed_pct90_league"),
        "Fouls won": fm("fotmob_fouls_won_league", "fotmob_fouls_won_per90_league", "fotmob_fouls_won_pct_league", "fotmob_fouls_won_pct90_league"),
    },
    "Defending": {
        "Defensive contributions": fm("fotmob_defensive_actions_league", "fotmob_defensive_actions_per90_league", "fotmob_defensive_actions_pct_league", "fotmob_defensive_actions_pct90_league"),
        "Tackles": fm("fotmob_tackles_league", "fotmob_tackles_per90_league", "fotmob_tackles_pct_league", "fotmob_tackles_pct90_league"),
        "Interceptions": fm("fotmob_interceptions_league", "fotmob_interceptions_per90_league", "fotmob_interceptions_pct_league", "fotmob_interceptions_pct90_league"),
        "Blocked shots": fm("fotmob_blocked_shots_league", "fotmob_blocked_shots_per90_league", "fotmob_blocked_shots_pct_league", "fotmob_blocked_shots_pct90_league"),
        "Fouls committed": fm("fotmob_fouls_league", "fotmob_fouls_per90_league", "fotmob_fouls_pct_league", "fotmob_fouls_pct90_league"),
        "Recoveries": fm("fotmob_recoveries_league", "fotmob_recoveries_per90_league", "fotmob_recoveries_pct_league", "fotmob_recoveries_pct90_league"),
        "Dribbled past": fm("fotmob_dribbled_past_league", "fotmob_dribbled_past_per90_league", "fotmob_dribbled_past_pct_league", "fotmob_dribbled_past_pct90_league"),
        "Clearances": fm("fotmob_clearances_league", "fotmob_clearances_per90_league", "fotmob_clearances_pct_league", "fotmob_clearances_pct90_league"),
        "Goals conceded while on pitch": fm("fotmob_goals_conceded_while_on_pitch_league", "fotmob_goals_conceded_while_on_pitch_per90_league", "fotmob_goals_conceded_while_on_pitch_pct_league", "fotmob_goals_conceded_while_on_pitch_pct90_league"),
        "xG against while on pitch": fm("fotmob_expected_goals_against_while_on_pitch_league", "fotmob_expected_goals_against_while_on_pitch_per90_league", "fotmob_expected_goals_against_while_on_pitch_pct_league", "fotmob_expected_goals_against_while_on_pitch_pct90_league"),
        "Error led to goal": fm("fotmob_error_led_to_goal_league", "fotmob_error_led_to_goal_per90_league", "fotmob_error_led_to_goal_pct_league", "fotmob_error_led_to_goal_pct90_league"),
    },
    "Discipline": {
        "Yellow cards": fm("fotmob_yellow_cards_league", "fotmob_yellow_cards_per90_league", "fotmob_yellow_cards_pct_league", "fotmob_yellow_cards_pct90_league"),
        "Red cards": fm("fotmob_red_cards_league", "fotmob_red_cards_per90_league", "fotmob_red_cards_pct_league", "fotmob_red_cards_pct90_league"),
    },
}

CATEGORY_ORDER = [
    "Shooting",
    "Passing",
    "Possession",
    "Defending",
    "Discipline",  # always rendered last
]

LEAGUE_GK_METRIC_CATALOG_RAW = {
    "Goalkeeping": {
        "Saves": fm("fotmob_saves_league", "fotmob_saves_per90_league", "fotmob_saves_pct_league", "fotmob_saves_pct90_league"),
        "Save percentage": fm("fotmob_save_percentage_league", "fotmob_save_percentage_per90_league", "fotmob_save_percentage_pct_league", "fotmob_save_percentage_pct90_league", True),
        "Goals conceded": fm("fotmob_goals_conceded_league", "fotmob_goals_conceded_per90_league", "fotmob_goals_conceded_pct_league", "fotmob_goals_conceded_pct90_league"),
        "Goals prevented": fm("fotmob_goals_prevented_league", "fotmob_goals_prevented_per90_league", "fotmob_goals_prevented_pct_league", "fotmob_goals_prevented_pct90_league"),
        "Clean sheets": fm("fotmob_clean_sheets_league", "fotmob_clean_sheets_per90_league", "fotmob_clean_sheets_pct_league", "fotmob_clean_sheets_pct90_league"),
        "Penalty saves": fm("fotmob_penalty_saves_league", "fotmob_penalty_saves_per90_league", "fotmob_penalty_saves_pct_league", "fotmob_penalty_saves_pct90_league"),
        "Penalty goals conceded": fm("fotmob_penalty_goals_conceded_league", "fotmob_penalty_goals_conceded_per90_league", "fotmob_penalty_goals_conceded_pct_league", "fotmob_penalty_goals_conceded_pct90_league"),
        "Penalty save %": fm("fotmob_penalty_save_percent_league", "fotmob_penalty_save_percent_per90_league", "fotmob_penalty_save_percent_pct_league", "fotmob_penalty_save_percent_pct90_league", True),
        "Error led to goal": fm("fotmob_error_led_to_goal_league", "fotmob_error_led_to_goal_per90_league", "fotmob_error_led_to_goal_pct_league", "fotmob_error_led_to_goal_pct90_league"),
        "Acted as sweeper": fm("fotmob_keeper_sweeper_league", "fotmob_keeper_sweeper_per90_league", "fotmob_keeper_sweeper_pct_league", "fotmob_keeper_sweeper_pct90_league"),
        "High claim": fm("fotmob_keeper_high_claim_league", "fotmob_keeper_high_claim_per90_league", "fotmob_keeper_high_claim_pct_league", "fotmob_keeper_high_claim_pct90_league"),
    },
    "Distribution": {
        "Successful passes": fm("fotmob_successful_passes_league", "fotmob_successful_passes_per90_league", "fotmob_successful_passes_pct_league", "fotmob_successful_passes_pct90_league"),
        "Successful passes %": fm("fotmob_successful_passes_accuracy_league", "fotmob_successful_passes_accuracy_per90_league", "fotmob_successful_passes_accuracy_pct_league", "fotmob_successful_passes_accuracy_pct90_league", True),
        "Accurate long balls": fm("fotmob_long_balls_accurate_league", "fotmob_long_balls_accurate_per90_league", "fotmob_long_balls_accurate_pct_league", "fotmob_long_balls_accurate_pct90_league"),
        "Accurate long balls %": fm("fotmob_long_ball_succeeeded_accuracy_league", "fotmob_long_ball_succeeeded_accuracy_per90_league", "fotmob_long_ball_succeeeded_accuracy_pct_league", "fotmob_long_ball_succeeeded_accuracy_pct90_league", True),
        "Expected assists (xA)": fm("fotmob_xa_league", "fotmob_expected_assists_per90_league", "fotmob_expected_assists_pct_league", "fotmob_expected_assists_pct90_league"),
    },
    "Discipline": {
        "Yellow cards": fm("fotmob_yellow_cards_league", "fotmob_yellow_cards_per90_league", "fotmob_yellow_cards_pct_league", "fotmob_yellow_cards_pct90_league"),
        "Red cards": fm("fotmob_red_cards_league", "fotmob_red_cards_per90_league", "fotmob_red_cards_pct_league", "fotmob_red_cards_pct90_league"),
    },
}

GK_CATEGORY_ORDER = [
    "Goalkeeping",
    "Distribution",
    "Discipline",  # always rendered last
]

# ============================================================
# STYLES
# ============================================================
S = {
    "page": {
        "background": f"radial-gradient(circle at top left, rgba(212,0,0,.18), transparent 28%), linear-gradient(135deg, {DBU_BG} 0%, #120609 45%, #09090b 100%)",
        "color": DBU_TEXT,
        "minHeight": "100vh",
        "fontFamily": "Inter, system-ui, sans-serif",
    },
    "shell": {"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "20px", "padding": "20px"},
    "sidebar": {
        "background": f"linear-gradient(180deg, {DBU_PANEL} 0%, #12090b 100%)",
        "border": f"1px solid {DBU_BORDER}",
        "borderRadius": "24px",
        "padding": "22px",
        "height": "calc(100vh - 40px)",
        "overflowY": "auto",
        "position": "sticky",
        "top": "20px",
        "boxShadow": "0 25px 50px rgba(0,0,0,.45)",
    },
    "main": {"display": "grid", "gap": "20px"},
    "panel": {
        "background": f"linear-gradient(180deg, {DBU_PANEL} 0%, {DBU_PANEL_2} 100%)",
        "border": f"1px solid {DBU_BORDER}",
        "borderRadius": "24px",
        "padding": "20px",
        "boxShadow": "0 18px 40px rgba(0,0,0,.32)",
    },
    "label": {"fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "0.16em", "color": DBU_RED_SOFT, "fontWeight": "800", "margin": "22px 0 10px"},
    "title": {"fontSize": "30px", "fontWeight": "900", "margin": "0 0 4px", "color": DBU_TEXT},
    "subtitle": {"color": DBU_MUTED, "margin": "0", "fontSize": "14px"},
    "cards": {"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(250px, 1fr))", "gap": "16px"},
    "card": {"background": f"linear-gradient(180deg, {DBU_PANEL_2} 0%, #1b0f13 100%)", "border": f"1px solid {DBU_BORDER}", "borderRadius": "18px", "padding": "15px", "cursor": "pointer", "transition": "all .18s ease"},
    "pill": {"display": "inline-block", "padding": "4px 10px", "borderRadius": "999px", "fontSize": "11px", "background": DBU_RED, "color": DBU_TEXT, "marginRight": "6px", "fontWeight": "700", "border": "1px solid rgba(255,255,255,.08)"},
}

# ============================================================
# DATA HELPERS
# ============================================================
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "player_id" not in df.columns:
        df["player_id"] = df.index.astype(str)

    # Stable key for selected-player state.
    # Prefer player_id, but fall back to identity fields if player_id is missing/null.
    def build_player_key(row: pd.Series) -> str:
        player_id = row.get("player_id", np.nan)
        if pd.notna(player_id) and str(player_id).strip() not in {"", "nan", "None"}:
            return str(player_id)

        name = str(row.get("player_name", "")).strip()
        team = str(row.get("team_name_unified", row.get("current_team", ""))).strip()
        age = str(row.get("age", "")).strip()
        return f"{name}__{team}__{age}"

    df["player_key"] = df.apply(build_player_key, axis=1)

    if "position_role_detailed" in df.columns:
        df["position_display"] = df["position_role_detailed"].replace(POSITION_DISPLAY_MAP)
    else:
        df["position_display"] = np.nan
    return df


def filter_existing_catalog(catalog: dict, df: pd.DataFrame) -> dict:
    clean_catalog = {}
    for category, metrics in catalog.items():
        for metric_name, type_map in metrics.items():
            existing_map = {}
            for key, col in type_map.items():
                if key == "is_percentage":
                    existing_map[key] = col
                elif col in df.columns:
                    existing_map[key] = col
            if any(k in existing_map for k in ["value", "per90"]):
                clean_catalog.setdefault(category, {})
                clean_catalog[category][metric_name] = existing_map
    return clean_catalog


def normalize_wyscout_category(category: str) -> str:
    category = str(category).strip().upper()
    if category == "OFFENSIVE DUELS":
        return "POSSESSION / OFFENSIVE DUELS"
    return category


def normalize_wyscout_type(metric_type: str) -> str:
    metric_type = str(metric_type).strip().lower()
    if metric_type in {"per 90", "per90", "p90"}:
        return "per90"
    if metric_type in {"%", "percent", "percentage"}:
        return "percentage"
    return "value"


def normalize_wyscout_profile(profile: str) -> str:
    profile = str(profile).strip().upper()
    if profile in {"GK", "GOALKEEPER", "GOALKEEPERS"}:
        return "GK"
    if profile in {"OUTFIELD", "FIELD", "PLAYER"}:
        return "OUTFIELD"
    if profile in {"BOTH", "ALL", "ANY", ""}:
        return "BOTH"
    return profile


def load_wyscout_metric_catalog(path: Path, df: pd.DataFrame) -> dict:
    catalog_df = pd.read_excel(path, sheet_name="ALL METRICS")
    required_cols = {"METRIC NAME", "METRIC LABEL", "CATEGORY", "TYPE", "PROFILE"}
    missing = required_cols - set(catalog_df.columns)
    if missing:
        raise ValueError(f"Missing columns in Wyscout catalog: {missing}")

    catalog = {}
    for _, row in catalog_df.iterrows():
        col = row.get("METRIC NAME")
        label = row.get("METRIC LABEL")
        category = row.get("CATEGORY")
        metric_type = row.get("TYPE")
        profile = row.get("PROFILE")

        if pd.isna(col) or pd.isna(label) or pd.isna(category) or pd.isna(metric_type):
            continue

        col = str(col).strip()
        label = str(label).strip()
        category = normalize_wyscout_category(category)
        metric_type = normalize_wyscout_type(metric_type)
        profile = normalize_wyscout_profile(profile) if pd.notna(profile) else "BOTH"

        if category == "PLAYER INFO":
            continue
        if col not in df.columns:
            continue

        catalog.setdefault(category, {})
        catalog[category].setdefault(label, {"profiles": set()})
        catalog[category][label][metric_type] = col
        catalog[category][label]["profiles"].add(profile)

    return catalog


# ============================================================
# ROLE-FIT / ARCHETYPE HELPERS
# ============================================================
def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def normalize_role_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_role_template_tables(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    templates = pd.read_excel(path, sheet_name="ROLE_METRIC_TEMPLATES")

    required_template_cols = {
        "position_group",
        "role_id",
        "role_name",
        "competence",
        "metric_name",
        "metric_label",
        "signal_type",
        "weight",
        "direction",
        "radar_use",
        "score_use",
        "modifier_tag",
        "interpretation",
        "warning",
    }
    missing_templates = required_template_cols - set(templates.columns)
    if missing_templates:
        raise ValueError(f"Missing columns in ROLE_METRIC_TEMPLATES: {missing_templates}")

    templates = templates.copy()
    templates["metric_name"] = templates["metric_name"].astype(str).str.strip()
    templates["role_id"] = templates["role_id"].astype(str).str.strip()
    templates["role_name"] = templates["role_name"].astype(str).str.strip()
    templates["position_group"] = templates["position_group"].astype(str).str.strip()
    templates["direction"] = templates["direction"].astype(str).str.strip().str.lower()
    templates["score_use"] = templates["score_use"].apply(parse_bool)
    templates["radar_use"] = templates["radar_use"].apply(parse_bool)
    templates["weight"] = pd.to_numeric(templates["weight"], errors="coerce").fillna(0)

    # v10 scoring references. These columns are required for raw-value scoring.
    for ref_col in ["reference_low", "reference_high"]:
        if ref_col not in templates.columns:
            templates[ref_col] = np.nan
        templates[ref_col] = pd.to_numeric(templates[ref_col], errors="coerce")

    if "use_percentile" not in templates.columns:
        templates["use_percentile"] = False
    templates["use_percentile"] = templates["use_percentile"].apply(parse_bool)

    if "scoring_mode" not in templates.columns:
        templates["scoring_mode"] = "raw_reference"

    try:
        role_map = pd.read_excel(path, sheet_name="POSITION_ROLE_MAP")
        role_map = role_map.copy()
        for col in role_map.columns:
            role_map[col] = role_map[col].apply(normalize_role_text)
    except Exception:
        role_map = pd.DataFrame(columns=["position", "position_group", "role_id", "role_name"])

    try:
        radar_ref = pd.read_excel(path, sheet_name="RADAR_REFERENCE")
        radar_ref = radar_ref.copy()
        for col in radar_ref.columns:
            if col != "radar_order":
                radar_ref[col] = radar_ref[col].apply(normalize_role_text)
        if "radar_order" in radar_ref.columns:
            radar_ref["radar_order"] = pd.to_numeric(radar_ref["radar_order"], errors="coerce").fillna(999)
        for ref_col in ["reference_low", "reference_high"]:
            if ref_col not in radar_ref.columns:
                radar_ref[ref_col] = np.nan
            radar_ref[ref_col] = pd.to_numeric(radar_ref[ref_col], errors="coerce")
        if "use_percentile" not in radar_ref.columns:
            radar_ref["use_percentile"] = False
        radar_ref["use_percentile"] = radar_ref["use_percentile"].apply(parse_bool)
        if "scoring_mode" not in radar_ref.columns:
            radar_ref["scoring_mode"] = "raw_reference"
    except Exception:
        radar_ref = pd.DataFrame(
            columns=[
                "position_group",
                "role_id",
                "role_name",
                "radar_name",
                "radar_order",
                "radar_label",
                "metric_name",
                "metric_label",
                "category",
                "type",
                "direction",
                "display_mode",
                "notes",
            ]
        )

    # Keep templates/radar rows as config, but mark availability against Wyscout columns.
    # Height exists in the parquet but must never enter role-fit or role radar.
    templates = templates[~templates["metric_name"].astype(str).str.lower().str.contains("height", na=False)].copy()
    templates["metric_available"] = templates["metric_name"].isin(df.columns)
    if "metric_name" in radar_ref.columns:
        radar_ref = radar_ref[~radar_ref["metric_name"].astype(str).str.lower().str.contains("height", na=False)].copy()
        radar_ref["metric_available"] = radar_ref["metric_name"].isin(df.columns)

    return templates, role_map, radar_ref


def get_player_position_group(row: pd.Series) -> str:
    return normalize_role_text(row.get("position_group_unified", ""))


def get_eligible_roles(row: pd.Series) -> pd.DataFrame:
    player_group = get_player_position_group(row)

    if not ROLE_MAP.empty and "position_group" in ROLE_MAP.columns and "role_id" in ROLE_MAP.columns:
        eligible = ROLE_MAP[ROLE_MAP["position_group"].astype(str).str.strip() == player_group].copy()
        if not eligible.empty:
            if "role_name" not in eligible.columns:
                names = ROLE_TEMPLATES[["role_id", "role_name"]].drop_duplicates()
                eligible = eligible.merge(names, on="role_id", how="left")
            return eligible[["role_id", "role_name"]].drop_duplicates()

    fallback = ROLE_TEMPLATES[ROLE_TEMPLATES["position_group"] == player_group]
    return fallback[["role_id", "role_name"]].drop_duplicates()


def contextual_metric_score(df: pd.DataFrame, row: pd.Series, metric_name: str, direction: str) -> Optional[float]:
    if metric_name not in df.columns or metric_name not in row.index:
        return None

    value = pd.to_numeric(pd.Series([row.get(metric_name)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None

    player_group = get_player_position_group(row)
    context = df.copy()

    if player_group and "position_group_unified" in context.columns:
        context = context[context["position_group_unified"].astype(str).str.strip() == player_group]

    values = pd.to_numeric(context[metric_name], errors="coerce").dropna()
    if len(values) < 5:
        values = pd.to_numeric(df[metric_name], errors="coerce").dropna()

    if values.empty:
        return None

    if direction == "lower":
        score = (values >= value).mean() * 100
    else:
        score = (values <= value).mean() * 100

    return float(np.clip(score, 0, 100))


def calculate_role_fit_scores(df: pd.DataFrame, row: pd.Series) -> list[dict]:
    eligible_roles = get_eligible_roles(row)
    role_results = []

    for _, role in eligible_roles.iterrows():
        role_id = str(role.get("role_id", "")).strip()
        role_name = str(role.get("role_name", role_id)).strip()

        role_metrics = ROLE_TEMPLATES[
            (ROLE_TEMPLATES["role_id"] == role_id)
            & (ROLE_TEMPLATES["score_use"])
            & (ROLE_TEMPLATES["weight"] > 0)
        ].copy()

        weighted_sum = 0.0
        total_weight = 0.0
        used_metrics = []
        modifier_scores = {}

        for _, metric in role_metrics.iterrows():
            metric_name = metric["metric_name"]
            weight = float(metric["weight"])
            direction = str(metric.get("direction", "higher")).strip().lower()
            score = contextual_metric_score(df, row, metric_name, direction)

            if score is None:
                continue

            weighted_sum += score * weight
            total_weight += weight

            metric_label = str(metric.get("metric_label", metric_name)).strip()
            raw_value = pd.to_numeric(pd.Series([wyscout_row.get(metric_name)]), errors="coerce").iloc[0]
            used_metrics.append(
                {
                    "metric_name": metric_name,
                    "metric_label": metric_label,
                    "raw_value": raw_value,
                    "signal_type": str(metric.get("signal_type", "")).strip().lower(),
                    "reference_low": reference_low,
                    "reference_high": reference_high,
                    "metric_score": score,
                    "score": score,
                    "weight": weight,
                    "weighted_score": score * weight,
                    "competence": str(metric.get("competence", "")).strip(),
                    "interpretation": str(metric.get("interpretation", "")).strip(),
                    "warning": str(metric.get("warning", "")).strip(),
                }
            )

            modifier_tag = str(metric.get("modifier_tag", "")).strip()
            if modifier_tag and modifier_tag.lower() not in {"nan", "none", ""}:
                modifier_scores.setdefault(modifier_tag, {"weighted_sum": 0.0, "weight": 0.0})
                modifier_scores[modifier_tag]["weighted_sum"] += score * weight
                modifier_scores[modifier_tag]["weight"] += weight

        if total_weight <= 0:
            continue

        role_score = weighted_sum / total_weight

        modifiers = []
        for tag, values in modifier_scores.items():
            if values["weight"] <= 0:
                continue
            tag_score = values["weighted_sum"] / values["weight"]
            if tag_score >= 60:
                modifiers.append({"tag": tag, "score": tag_score})

        modifiers = sorted(modifiers, key=lambda x: x["score"], reverse=True)[:4]
        used_metrics = sorted(used_metrics, key=lambda x: x["score"] * x["weight"], reverse=True)

        role_results.append(
            {
                "role_id": role_id,
                "role_name": role_name,
                "role_fit_score": role_score,
                "modifiers": modifiers,
                "used_metrics": used_metrics,
            }
        )

    return sorted(role_results, key=lambda x: x["role_fit_score"], reverse=True)


DF = load_dataset(DATA_PATH)
LEAGUE_METRIC_CATALOG = filter_existing_catalog(LEAGUE_METRIC_CATALOG_RAW, DF)
LEAGUE_GK_METRIC_CATALOG = filter_existing_catalog(LEAGUE_GK_METRIC_CATALOG_RAW, DF)
WYSCOUT_METRIC_CATALOG = load_wyscout_metric_catalog(WYSCOUT_CATALOG_PATH, DF)
TOTAL_SEASON_DF = DF[DF["has_wyscout"].fillna(False).astype(bool)].copy() if "has_wyscout" in DF.columns else DF.copy()
ROLE_TEMPLATES, ROLE_MAP, RADAR_REFERENCE = load_role_template_tables(ROLE_TEMPLATES_PATH, TOTAL_SEASON_DF)

age_min = int(DF["age"].min()) if "age" in DF.columns else 15
age_max = int(DF["age"].max()) if "age" in DF.columns else 40

# ============================================================
# FORMAT + FILTER HELPERS
# ============================================================
def format_value(value, decimals: int = 2):
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return str(int(value)) if value.is_integer() else f"{value:.{decimals}f}"
    return str(value)


def format_percentage(value):
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) <= 1:
        value *= 100
    return f"{value:.1f}%"


def format_percentile(value):
    if pd.isna(value):
        return "—"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def expand_pitch_positions(pitch_positions: List[str]) -> List[str]:
    expanded = []
    for pos in pitch_positions or []:
        expanded.extend(PITCH_FILTER_MAP.get(pos, [pos]))
    return list(dict.fromkeys(expanded))


def filter_df(df, position_details, leagues, teams, age_range, minutes_range, search_text):
    dff = df.copy()
    if position_details and "position_role_detailed" in dff.columns:
        dff = dff[dff["position_role_detailed"].isin(position_details)]
    if leagues and "league_unified" in dff.columns:
        dff = dff[dff["league_unified"].isin(leagues)]
    if teams and "team_name_unified" in dff.columns:
        dff = dff[dff["team_name_unified"].isin(teams)]
    if age_range and "age" in dff.columns:
        dff = dff[dff["age"].between(age_range[0], age_range[1], inclusive="both")]
    if minutes_range and "minutes_played" in dff.columns:
        dff = dff[dff["minutes_played"].fillna(0).between(minutes_range[0], minutes_range[1], inclusive="both")]
    if search_text and "player_name" in dff.columns:
        dff = dff[dff["player_name"].astype(str).str.contains(search_text, case=False, na=False)]
    return dff


def is_goalkeeper(row: pd.Series) -> bool:
    position_group = str(row.get("position_group_unified", "")).strip()
    position_role = str(row.get("position_role_unified", "")).strip()
    return position_group == "Goalkeepers" or position_role == "GK"


def wyscout_metric_allowed_for_player(type_map: dict, row: pd.Series) -> bool:
    """Use the Excel PROFILE column as the source of truth for Wyscout rendering."""
    profiles = type_map.get("profiles", {"BOTH"})

    if isinstance(profiles, str):
        profiles = {profiles}

    profiles = {normalize_wyscout_profile(p) for p in profiles if pd.notna(p)}

    if not profiles:
        profiles = {"BOTH"}

    if is_goalkeeper(row):
        return bool(profiles & {"GK", "BOTH"})

    return bool(profiles & {"OUTFIELD", "BOTH"})

# ============================================================
# UI COMPONENTS
# ============================================================
def pitch_button(role: str, selected: List[str]):
    active = role in (selected or [])
    return html.Button(
        role,
        id={"type": "pitch-role", "role": role},
        n_clicks=0,
        style={
            "width": "54px",
            "height": "54px",
            "borderRadius": "999px",
            "border": f"2px solid {DBU_RED_SOFT}" if active else "1px solid rgba(255,255,255,.12)",
            "background": f"linear-gradient(180deg, {DBU_RED} 0%, {DBU_RED_DARK} 100%)" if active else "linear-gradient(180deg,#131722 0%,#0d1118 100%)",
            "color": "white",
            "fontWeight": "800",
            "fontSize": "12px",
            "cursor": "pointer",
            "transition": "all .18s ease",
            "boxShadow": "0 0 24px rgba(212,0,0,.45)" if active else "0 6px 14px rgba(0,0,0,.30)",
        },
    )


def build_pitch_map(selected=None):
    selected = selected or []
    return html.Div(
        [
            html.Div("Position map", style={**S["label"], "marginTop": "24px"}),
            html.Div(
                [
                    html.Div(
                        [pitch_button(role, selected) for role in row],
                        style={"display": "flex", "justifyContent": "center", "gap": "12px", "margin": "10px 0"},
                    )
                    for row in PITCH_ROWS
                ],
                style={
                    "background": f"radial-gradient(circle at center, rgba(212,0,0,.18), transparent 55%), linear-gradient(180deg, {DBU_PITCH_TOP} 0%, {DBU_PITCH_MID} 55%, {DBU_PITCH_BOTTOM} 100%)",
                    "border": f"1px solid rgba(212,0,0,.55)",
                    "borderRadius": "22px",
                    "padding": "22px 12px",
                    "boxShadow": "inset 0 0 0 1px rgba(255,255,255,.04), 0 0 45px rgba(212,0,0,.12)",
                },
            ),
        ]
    )


def player_card(row: pd.Series, selected_player_id=None):
    player_id = str(row.get("player_key"))
    selected = selected_player_id is not None and player_id == str(selected_player_id)
    style = dict(S["card"])
    if selected:
        style.update({"border": f"2px solid {DBU_RED_SOFT}", "boxShadow": "0 0 26px rgba(212,0,0,.38)", "transform": "translateY(-2px)"})

    return html.Div(
        [
            html.Div([html.Span(row.get("position_display", row.get("position_role_detailed", "-")), style=S["pill"]), html.Span(str(row.get("league_unified", "-")), style=S["pill"])]),
            html.H3(str(row.get("player_name", "Unknown")), style={"margin": "12px 0 4px", "fontSize": "18px"}),
            html.Div(str(row.get("team_name_unified", "-")), style={"color": DBU_MUTED, "fontSize": "13px", "marginBottom": "14px"}),
            html.Div(
                [
                    html.Div([html.Div("Age", style={"color": DBU_MUTED, "fontSize": "11px"}), html.Strong(format_value(row.get("age", np.nan)))]),
                    html.Div([html.Div("Minutes", style={"color": DBU_MUTED, "fontSize": "11px"}), html.Strong(format_value(row.get("minutes_played", np.nan)))]),
                ],
                style={"display": "flex", "gap": "18px"},
            ),
        ],
        id={"type": "player-card", "player_id": player_id},
        n_clicks=0,
        style=style,
    )


def compact_selected_strip(row: Optional[pd.Series]):
    if row is None:
        return None
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Selected player", style={**S["label"], "marginTop": "0"}),
                    html.H3(row.get("player_name", "Unknown"), style={"margin": "0"}),
                    html.Div(f"{row.get('position_display', '-')} · {row.get('team_name_unified', '-')} · {row.get('league_unified', '-')}", style=S["subtitle"]),
                ]
            ),
            html.Button("Change player", id="clear_selected_player_btn", n_clicks=0, style={"background": DBU_RED, "color": DBU_TEXT, "border": "none", "borderRadius": "999px", "padding": "10px 16px", "fontWeight": "800", "cursor": "pointer"}),
        ],
        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "16px"},
    )


def compact_info_field(label, value, suffix: str = ""):
    display = format_value(value)
    if display != "—" and suffix:
        display = f"{display} {suffix}"

    return html.Div(
        [
            html.Div(label, style={"color": DBU_MUTED, "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": ".08em"}),
            html.Div(display, style={"fontSize": "14px", "fontWeight": "800", "marginTop": "2px"}),
        ],
        style={"minWidth": "88px"},
    )


def player_bio_card(row: pd.Series):
    name = row.get("player_name", "Unknown Player")
    team = row.get("team_name_unified", "-")
    league = row.get("league_unified", "-")
    country = row.get("league_country", "-")
    pos_detail = row.get("position_role_detailed", "-")
    pos_group = row.get("position_group_unified", "-")
    pos_role = row.get("position_role_unified", "-")
    league_text = league if pd.isna(country) or country == "-" else f"{league} · {country}"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "PHOTO",
                        style={
                            "width": "96px",
                            "height": "120px",
                            "borderRadius": "18px",
                            "background": "linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.02))",
                            "border": f"1px solid {DBU_BORDER}",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "color": DBU_MUTED,
                            "fontSize": "11px",
                            "fontWeight": "900",
                            "letterSpacing": ".12em",
                            "flexShrink": 0,
                        },
                    ),
                    html.Div(
                        [
                            html.Div("Player profile", style={**S["label"], "margin": "0 0 8px"}),
                            html.H2(str(name), style={"margin": "0 0 6px", "fontSize": "32px", "letterSpacing": "-.03em"}),
                            html.Div(f"{pos_detail} · {team} · {league_text}", style=S["subtitle"]),
                            html.Div(
                                [
                                    html.Span(pos_group, style=S["pill"]),
                                    html.Span(pos_role, style=S["pill"]),
                                    html.Span(pos_detail, style=S["pill"]),
                                ],
                                style={"marginTop": "12px"},
                            ),
                        ],
                        style={"minWidth": "240px", "maxWidth": "420px", "flex": "0 1 420px"},
                    ),
                    html.Div(
                        [
                            role_radar_component(row),
                            similar_players_panel(row),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "300px minmax(520px, 1fr)",
                            "gap": "12px",
                            "alignItems": "stretch",
                            "flex": "1 1 820px",
                            "minWidth": "620px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "18px", "alignItems": "center", "justifyContent": "flex-start", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    compact_info_field("Age", row.get("age", np.nan)),
                    compact_info_field("Height", row.get("height_cm", np.nan), "cm"),
                    compact_info_field("Weight", row.get("weight_kg", np.nan), "kg"),
                    compact_info_field("Foot", row.get("preferred_foot", "-")),
                    compact_info_field("Nationality", row.get("nationality", "-")),
                    compact_info_field("Minutes", row.get("minutes_played", np.nan)),
                    compact_info_field("Matches", row.get("matches_played", np.nan)),
                    compact_info_field("Starts", row.get("starts", np.nan)),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(92px, 1fr))",
                    "gap": "12px",
                    "marginTop": "18px",
                    "paddingTop": "16px",
                    "borderTop": f"1px solid {DBU_BORDER}",
                },
            ),
        ],
        style={
            "background": "linear-gradient(135deg, rgba(212,0,0,.13), rgba(255,255,255,.035))",
            "border": f"1px solid {DBU_BORDER}",
            "borderRadius": "22px",
            "padding": "18px",
            "boxShadow": "inset 0 0 0 1px rgba(255,255,255,.025)",
        },
    )


def percentile_color(value):
    if pd.isna(value):
        return DBU_BORDER
    value = float(value)
    if value < 25:
        return "#ef4444"
    if value < 50:
        return "#f97316"
    if value < 75:
        return "#eab308"
    return "#22c55e"


def percentile_bar(value):
    if pd.isna(value):
        width = 0
        label = "—"
        color = DBU_BORDER
    else:
        width = max(0, min(100, float(value)))
        label = format_percentile(value)
        color = percentile_color(value)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        style={
                            "width": f"{width}%",
                            "height": "100%",
                            "borderRadius": "999px",
                            "background": color,
                        }
                    )
                ],
                style={
                    "height": "6px",
                    "background": "rgba(255,255,255,.08)",
                    "borderRadius": "999px",
                    "overflow": "hidden",
                    "minWidth": "90px",
                    "flex": "1",
                },
            ),
            html.Div(label, style={"fontSize": "11px", "color": DBU_MUTED, "width": "34px", "textAlign": "right"}),
        ],
        style={"display": "flex", "gap": "8px", "alignItems": "center", "minWidth": "130px"},
    )


def metric_list_row(metric_name: str, display_value: str, rank_value=None, value_label: str = ""):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        metric_name,
                        style={
                            "fontSize": "14px",
                            "fontWeight": "850",
                            "color": DBU_TEXT,
                            "lineHeight": "1.15",
                            "minHeight": "30px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                display_value,
                                style={
                                    "fontSize": "26px",
                                    "fontWeight": "950",
                                    "letterSpacing": "-.03em",
                                    "lineHeight": "1",
                                },
                            ),
                            html.Div(
                                value_label,
                                style={
                                    "fontSize": "9px",
                                    "color": DBU_MUTED,
                                    "textTransform": "uppercase",
                                    "letterSpacing": ".08em",
                                    "marginTop": "3px",
                                },
                            ) if value_label else None,
                        ],
                        style={"textAlign": "right", "marginLeft": "10px", "flexShrink": 0},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "gap": "10px",
                    "alignItems": "flex-start",
                },
            ),
            html.Div(
                percentile_bar(rank_value) if rank_value is not None else None,
                style={"marginTop": "7px"} if rank_value is not None else {"display": "none"},
            ),
        ],
        style={
            "background": "rgba(255,255,255,.04)",
            "border": f"1px solid {DBU_BORDER}",
            "borderRadius": "14px",
            "padding": "9px 11px",
            "minHeight": "74px",
            "boxShadow": "inset 0 0 0 1px rgba(255,255,255,.018)",
        },
    )


def league_metric_row(metric_name: str, type_map: dict, row: pd.Series, metric_mode: str):
    if metric_mode == "total":
        value_col = type_map.get("value")
        rank_col = type_map.get("percentile")
        value_label = "Total"
    else:
        value_col = type_map.get("per90")
        rank_col = type_map.get("percentile_per90")
        value_label = "Per 90"

    if value_col is None:
        return None

    value = row.get(value_col, np.nan)
    rank = row.get(rank_col, np.nan) if rank_col else np.nan
    display_value = format_percentage(value) if type_map.get("is_percentage") else format_value(value)

    return metric_list_row(metric_name, display_value, rank, value_label)


def wyscout_metric_row(metric_name: str, type_map: dict, row: pd.Series, metric_mode: str):
    if not wyscout_metric_allowed_for_player(type_map, row):
        return None

    if metric_mode == "per90":
        value_col = type_map.get("per90")
        value_type = "per90"
        value_label = "Per 90"
    else:
        value_col = type_map.get("value")
        value_type = "value"
        value_label = "Total"

    if value_col is None and "percentage" in type_map:
        value_col = type_map.get("percentage")
        value_type = "percentage"
        value_label = "Percentage"

    if value_col is None:
        return None

    value = row.get(value_col, np.nan)
    display_value = format_percentage(value) if value_type == "percentage" else format_value(value)
    return metric_list_row(metric_name, display_value, None, value_label)


def metric_category_panel(category: str, metric_group: dict, row: pd.Series, metric_mode: str, source: str):
    rows = []
    for metric_name, type_map in metric_group.items():
        if source == "league":
            component = league_metric_row(metric_name, type_map, row, metric_mode)
        else:
            component = wyscout_metric_row(metric_name, type_map, row, metric_mode)
        if component is not None:
            rows.append(component)

    if not rows:
        return None

    return html.Div(
        [
            html.Div(category, style={
                **S["label"],
                "margin": "0 0 14px",
                "fontSize": "11px",
                "letterSpacing": ".18em",
                "fontWeight": "900",
            }),
            html.Div(
                rows,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "8px 10px",
                    "alignItems": "stretch",
                    "width": "100%",
                },
            ),
        ],
        style={
            "background": "rgba(0,0,0,.16)",
            "border": "1px solid rgba(255,255,255,.06)",
            "borderRadius": "20px",
            "padding": "14px",
            "boxShadow": "0 8px 18px rgba(0,0,0,.16)",
            "width": "100%",
        },
    )


def sort_category_panels_for_display(panels: list) -> list:
    """Keep Discipline visually last because it is low-priority and usually small."""
    discipline = []
    others = []

    for panel in panels:
        panel_text = str(panel)
        if "Discipline" in panel_text or "DISCIPLINE" in panel_text:
            discipline.append(panel)
        else:
            others.append(panel)

    return others + discipline


def category_masonry_layout(panels: list):
    """Two independent vertical columns so short categories do not create dead space."""
    panels = sort_category_panels_for_display(panels)

    left_col = []
    right_col = []

    for idx, panel in enumerate(panels):
        if idx % 2 == 0:
            left_col.append(panel)
        else:
            right_col.append(panel)

    return html.Div(
        [
            html.Div(left_col, style={"display": "flex", "flexDirection": "column", "gap": "18px", "minWidth": 0}),
            html.Div(right_col, style={"display": "flex", "flexDirection": "column", "gap": "18px", "minWidth": 0}),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "18px",
            "marginTop": "18px",
            "alignItems": "start",
        },
    )


def league_profile_metrics(row: pd.Series, metric_mode: str):
    if is_goalkeeper(row):
        catalog = LEAGUE_GK_METRIC_CATALOG
        category_order = GK_CATEGORY_ORDER
    else:
        catalog = LEAGUE_METRIC_CATALOG
        category_order = CATEGORY_ORDER

    panels = []
    for category in category_order:
        panel = metric_category_panel(category, catalog.get(category, {}), row, metric_mode, source="league")
        if panel is not None:
            panels.append(panel)

    if not panels:
        return html.Div("No League only metrics available.", style={"color": DBU_MUTED, "padding": "20px"})
    return category_masonry_layout(panels)


def wyscout_profile_metrics(row: pd.Series, metric_mode: str):
    panels = []

    for category in sorted(WYSCOUT_METRIC_CATALOG.keys()):
        metric_group = WYSCOUT_METRIC_CATALOG.get(category, {})

        # The row-level filtering happens inside wyscout_metric_row via PROFILE.
        # Empty panels are automatically skipped.
        panel = metric_category_panel(category, metric_group, row, metric_mode, source="total")

        if panel is not None:
            panels.append(panel)

    if not panels:
        return html.Div("No Total season metrics available.", style={"color": DBU_MUTED, "padding": "20px"})

    return category_masonry_layout(panels)


# ============================================================
# ROLE-FIT OVERRIDES: Wyscout Total Season only
# ============================================================
def get_wyscout_player_row(row: pd.Series) -> Optional[pd.Series]:
    """Find the selected player in the Wyscout/Total season universe."""
    if TOTAL_SEASON_DF.empty:
        return None

    if "player_key" in TOTAL_SEASON_DF.columns and "player_key" in row.index:
        match = TOTAL_SEASON_DF[TOTAL_SEASON_DF["player_key"].astype(str) == str(row.get("player_key"))]
        if not match.empty:
            return match.iloc[0]

    if "player_id" in TOTAL_SEASON_DF.columns and "player_id" in row.index:
        player_id = row.get("player_id")
        if pd.notna(player_id):
            match = TOTAL_SEASON_DF[TOTAL_SEASON_DF["player_id"].astype(str) == str(player_id)]
            if not match.empty:
                return match.iloc[0]

    name = str(row.get("player_name", "")).strip().lower()
    team = str(row.get("team_name_unified", row.get("current_team", ""))).strip().lower()

    if name and "player_name" in TOTAL_SEASON_DF.columns:
        match = TOTAL_SEASON_DF[TOTAL_SEASON_DF["player_name"].astype(str).str.strip().str.lower() == name]
        if team and "team_name_unified" in TOTAL_SEASON_DF.columns:
            team_match = match[match["team_name_unified"].astype(str).str.strip().str.lower() == team]
            if not team_match.empty:
                return team_match.iloc[0]
        if not match.empty:
            return match.iloc[0]

    return None


POSITION_ALIAS_MAP = {
    "LW": "LW",
    "RW": "RW",
    "LWF": "LW",
    "RWF": "RW",
    "LF": "LW",
    "RF": "RW",
    "WF": "WF",
    "AML": "LW",
    "AMR": "RW",
    "LMF": "LMF",
    "RMF": "RMF",
    "LWM": "LW",
    "RWM": "RW",
    "LEFT WINGER": "LW",
    "RIGHT WINGER": "RW",
    "LEFT FORWARD": "LW",
    "RIGHT FORWARD": "RW",
    "ST": "CF",
    "STRIKER": "CF",
    "CENTER FORWARD": "CF",
    "CENTRE FORWARD": "CF",
    "DM": "DMF",
    "CM": "CMF",
    "AM": "AMF",
}


def normalize_position_code(value) -> str:
    if pd.isna(value):
        return ""
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE"}:
        return ""
    return POSITION_ALIAS_MAP.get(code, code)


def get_normalized_position(row: pd.Series) -> str:
    for col in ["position_role_detailed", "wyscout_Position_total", "position_display", "position_role_unified", "primary_position_clean"]:
        if col in row.index and pd.notna(row.get(col)):
            value = normalize_position_code(row.get(col))
            if value:
                return value
    return ""


def get_position_map_column() -> Optional[str]:
    for col in ["position_code", "position", "normalized_position", "position_role_detailed", "position_detail"]:
        if col in ROLE_MAP.columns:
            return col
    return None


def get_role_id_column() -> Optional[str]:
    for col in ["role_id", "role_ids", "eligible_role_ids"]:
        if col in ROLE_MAP.columns:
            return col
    return None


def split_role_ids(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).replace(";", ",").replace("|", ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def get_eligible_roles(row: pd.Series) -> pd.DataFrame:
    player_name = row.get("player_name", "Unknown")
    raw_position = row.get("wyscout_Position_total", row.get("position_role_detailed", ""))
    normalized_position = get_normalized_position(row)
    position_col = get_position_map_column()
    role_col = get_role_id_column()

    if not normalized_position or ROLE_MAP.empty or position_col is None or role_col is None:
        print("[ROLE_MAP_DEBUG] PLAYER:", player_name)
        print("[ROLE_MAP_DEBUG] RAW POSITION:", raw_position)
        print("[ROLE_MAP_DEBUG] NORMALIZED POSITION:", normalized_position)
        print("[ROLE_MAP_DEBUG] No role map / position column / role column available")
        return pd.DataFrame(columns=["role_id", "role_name", "position_group"])

    map_positions = ROLE_MAP[position_col].apply(normalize_position_code)
    mapping_rows = ROLE_MAP[map_positions == normalized_position].copy()

    if mapping_rows.empty:
        print("[ROLE_MAP_DEBUG] PLAYER:", player_name)
        print("[ROLE_MAP_DEBUG] RAW POSITION:", raw_position)
        print("[ROLE_MAP_DEBUG] NORMALIZED POSITION:", normalized_position)
        print("[ROLE_MAP_DEBUG] POSITION_ROLE_MAP positions:", sorted(map_positions.dropna().unique().tolist()))
        print("[ROLE_MAP_DEBUG] ELIGIBLE ROLE MAP: EMPTY")
        return pd.DataFrame(columns=["role_id", "role_name", "position_group"])

    print("[ROLE_MAP_DEBUG] PLAYER:", player_name)
    print("[ROLE_MAP_DEBUG] RAW POSITION:", raw_position)
    print("[ROLE_MAP_DEBUG] NORMALIZED POSITION:", normalized_position)
    print("[ROLE_MAP_DEBUG] ELIGIBLE ROLE MAP:", mapping_rows.to_dict("records"))

    role_rows = []
    for _, map_row in mapping_rows.iterrows():
        for role_id in split_role_ids(map_row.get(role_col)):
            role_name = ""
            if "role_name" in map_row.index and map_row.get("role_name"):
                role_name = str(map_row.get("role_name")).strip()
            if not role_name:
                role_names = ROLE_TEMPLATES[ROLE_TEMPLATES["role_id"] == role_id]["role_name"].dropna().unique()
                role_name = str(role_names[0]) if len(role_names) else role_id

            role_rows.append(
                {
                    "role_id": role_id,
                    "role_name": role_name,
                    "position_group": str(map_row.get("position_group", "")).strip(),
                }
            )

    out = pd.DataFrame(role_rows).drop_duplicates("role_id") if role_rows else pd.DataFrame(columns=["role_id", "role_name", "position_group"])
    print("[ROLE_MAP_DEBUG] ELIGIBLE ROLE IDS:", out["role_id"].dropna().unique().tolist() if not out.empty else [])
    return out


def get_role_context_df(row: pd.Series, position_group: str = "") -> pd.DataFrame:
    context = TOTAL_SEASON_DF.copy()
    normalized_position = get_normalized_position(row)

    comparable_groups = {
        "GK": ["GK"],
        "CB": ["CB", "LCB", "RCB"],
        "LCB": ["CB", "LCB", "RCB"],
        "RCB": ["CB", "LCB", "RCB"],
        "LB": ["LB", "RB", "LWB", "RWB"],
        "RB": ["LB", "RB", "LWB", "RWB"],
        "LWB": ["LB", "RB", "LWB", "RWB"],
        "RWB": ["LB", "RB", "LWB", "RWB"],
        "DMF": ["DMF", "LDMF", "RDMF", "DM"],
        "LDMF": ["DMF", "LDMF", "RDMF", "DM"],
        "RDMF": ["DMF", "LDMF", "RDMF", "DM"],
        "CMF": ["CMF", "LCMF", "RCMF", "CM"],
        "LCMF": ["CMF", "LCMF", "RCMF", "CM"],
        "RCMF": ["CMF", "LCMF", "RCMF", "CM"],
        "AMF": ["AMF", "LAMF", "RAMF", "AM"],
        "LAMF": ["AMF", "LAMF", "RAMF", "AM"],
        "RAMF": ["AMF", "LAMF", "RAMF", "AM"],
        "LW": ["LW", "RW", "LMF", "RMF", "LWF", "RWF", "LF", "RF", "AML", "AMR", "LWM", "RWM", "WF"],
        "RW": ["LW", "RW", "LMF", "RMF", "LWF", "RWF", "LF", "RF", "AML", "AMR", "LWM", "RWM", "WF"],
        "LMF": ["LW", "RW", "LMF", "RMF", "LWF", "RWF", "LF", "RF", "AML", "AMR", "LWM", "RWM", "WF"],
        "RMF": ["LW", "RW", "LMF", "RMF", "LWF", "RWF", "LF", "RF", "AML", "AMR", "LWM", "RWM", "WF"],
        "CF": ["CF", "ST", "STRIKER"],
    }

    if normalized_position and "position_role_detailed" in context.columns:
        details = context["position_role_detailed"].apply(normalize_position_code)
        family = comparable_groups.get(normalized_position, [normalized_position])
        same_family = context[details.isin(family)]
        if len(same_family) >= 10:
            return same_family

    if position_group and "position_group_unified" in context.columns:
        same_group = context[context["position_group_unified"].astype(str).str.strip() == position_group]
        if len(same_group) >= 10:
            return same_group

    return context


def raw_reference_metric_score(
    raw_value,
    reference_low,
    reference_high,
    direction: str = "higher",
) -> Optional[float]:
    """v10 role/radar scoring: raw value normalized against Excel references.

    No percentiles are used. Score is 0-100 based on reference_low/reference_high.
    If direction == lower, the score is inverted.
    """
    try:
        raw_value = float(raw_value)
        reference_low = float(reference_low)
        reference_high = float(reference_high)
    except Exception:
        return None

    if pd.isna(raw_value) or pd.isna(reference_low) or pd.isna(reference_high):
        return None

    if reference_high == reference_low:
        return None

    score01 = (raw_value - reference_low) / (reference_high - reference_low)
    score01 = max(0.0, min(1.0, score01))
    metric_score = score01 * 100

    if str(direction).strip().lower() == "lower":
        metric_score = 100 - metric_score

    return float(np.clip(metric_score, 0, 100))


def contextual_metric_score(
    df: pd.DataFrame,
    row: pd.Series,
    metric_name: str,
    direction: str,
    position_group: str = "",
    reference_low=np.nan,
    reference_high=np.nan,
) -> Optional[float]:
    """Compatibility wrapper for v10 raw-reference scoring."""
    if metric_name not in df.columns or metric_name not in row.index:
        return None

    raw_value = pd.to_numeric(pd.Series([row.get(metric_name)]), errors="coerce").iloc[0]
    if pd.isna(raw_value):
        return None

    return raw_reference_metric_score(raw_value, reference_low, reference_high, direction)


def calculate_role_fit_scores(df: pd.DataFrame, row: pd.Series) -> list[dict]:
    wyscout_row = get_wyscout_player_row(row)
    if wyscout_row is None:
        print("[ROLE_FIT_DEBUG] No Wyscout row found for:", row.get("player_name", "Unknown"))
        return []

    normalized_position = get_normalized_position(wyscout_row)
    eligible_roles = get_eligible_roles(wyscout_row)

    print("[ROLE_FIT_DEBUG] player=", wyscout_row.get("player_name", "Unknown"))
    print("[ROLE_FIT_DEBUG] normalized_position=", normalized_position)
    print("[ROLE_FIT_DEBUG] eligible_role_ids=", eligible_roles["role_id"].tolist() if not eligible_roles.empty else [])
    print("[ROLE_FIT_DEBUG] using_total_season_df_rows=", len(TOTAL_SEASON_DF))

    if eligible_roles.empty:
        print("[ROLE_FIT_DEBUG] No role mapping found for this position.")
        return []

    role_results = []

    for _, role in eligible_roles.iterrows():
        role_id = str(role.get("role_id", "")).strip()
        role_name = str(role.get("role_name", role_id)).strip()
        position_group = str(role.get("position_group", "")).strip()

        role_metrics_all = ROLE_TEMPLATES[ROLE_TEMPLATES["role_id"] == role_id].copy()
        role_metrics = role_metrics_all[
            (role_metrics_all["score_use"])
            & (role_metrics_all["weight"] > 0)
            & (role_metrics_all["metric_available"])
        ].copy()

        missing_metrics = sorted(set(role_metrics_all["metric_name"]) - set(TOTAL_SEASON_DF.columns))
        matched_columns = sorted(set(role_metrics["metric_name"]) & set(TOTAL_SEASON_DF.columns))

        print("[ROLE_FIT_DEBUG] role_id=", role_id)
        print("[ROLE_FIT_DEBUG] role_template_rows=", len(role_metrics_all))
        print("[ROLE_FIT_DEBUG] matching_columns=", len(matched_columns))
        print("[ROLE_FIT_DEBUG] missing_metrics=", missing_metrics[:12])

        if role_metrics_all.empty or role_metrics.empty:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        used_metrics = []
        modifier_scores = {}
        null_metrics = []

        for _, metric in role_metrics.iterrows():
            metric_name = metric["metric_name"]
            weight = float(metric["weight"])
            direction = str(metric.get("direction", "higher")).strip().lower()
            reference_low = metric.get("reference_low", np.nan)
            reference_high = metric.get("reference_high", np.nan)
            score = contextual_metric_score(
                TOTAL_SEASON_DF,
                wyscout_row,
                metric_name,
                direction,
                position_group,
                reference_low,
                reference_high,
            )

            if score is None:
                null_metrics.append(metric_name)
                continue

            weighted_sum += score * weight
            total_weight += weight

            metric_label = str(metric.get("metric_label", metric_name)).strip()
            used_metrics.append(
                {
                    "metric_name": metric_name,
                    "metric_label": metric_label,
                    "score": score,
                    "weight": weight,
                    "competence": str(metric.get("competence", "")).strip(),
                    "interpretation": str(metric.get("interpretation", "")).strip(),
                    "warning": str(metric.get("warning", "")).strip(),
                }
            )

            modifier_tag = str(metric.get("modifier_tag", "")).strip()
            if modifier_tag and modifier_tag.lower() not in {"nan", "none", ""}:
                modifier_scores.setdefault(modifier_tag, {"weighted_sum": 0.0, "weight": 0.0})
                modifier_scores[modifier_tag]["weighted_sum"] += score * weight
                modifier_scores[modifier_tag]["weight"] += weight

        if total_weight <= 0:
            print("[ROLE_FIT_DEBUG] Role templates matched, but player has no valid numeric values for role:", role_id)
            continue

        role_score = weighted_sum / total_weight
        modifiers = []

        for tag, values in modifier_scores.items():
            if values["weight"] <= 0:
                continue
            tag_score = values["weighted_sum"] / values["weight"]
            if tag_score >= 60:
                modifiers.append({"tag": tag, "score": tag_score})

        role_results.append(
            {
                "role_id": role_id,
                "role_name": role_name,
                "role_fit_score": role_score,
                "modifiers": sorted(modifiers, key=lambda x: x["score"], reverse=True)[:4],
                "used_metrics": sorted(used_metrics, key=lambda x: x["score"] * x["weight"], reverse=True),
                "missing_metrics": missing_metrics,
                "null_metrics": null_metrics,
            }
        )

    return sorted(role_results, key=lambda x: x["role_fit_score"], reverse=True)


def get_best_role_result(row: pd.Series) -> Optional[dict]:
    scores = calculate_role_fit_scores(TOTAL_SEASON_DF, row)
    return scores[0] if scores else None


def get_role_radar_rows(role_id: str) -> pd.DataFrame:
    if not RADAR_REFERENCE.empty and "role_id" in RADAR_REFERENCE.columns:
        rows = RADAR_REFERENCE[RADAR_REFERENCE["role_id"].astype(str).str.strip() == str(role_id)].copy()
        rows = rows[rows["metric_name"].isin(TOTAL_SEASON_DF.columns)]
        if not rows.empty:
            return rows.sort_values("radar_order") if "radar_order" in rows.columns else rows

    fallback = ROLE_TEMPLATES[
        (ROLE_TEMPLATES["role_id"] == str(role_id))
        & (ROLE_TEMPLATES["radar_use"])
        & (ROLE_TEMPLATES["metric_available"])
    ].copy()

    if fallback.empty:
        return pd.DataFrame()

    fallback["radar_label"] = fallback["metric_label"]
    fallback["radar_order"] = range(1, len(fallback) + 1)
    return fallback


def build_role_radar_figure(row: pd.Series, role_result: Optional[dict]):
    fig = go.Figure()

    wyscout_row = get_wyscout_player_row(row)
    if wyscout_row is None or role_result is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No Wyscout role radar available", showarrow=False, font=dict(color=DBU_MUTED))],
            margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig

    radar_rows = get_role_radar_rows(role_result["role_id"])
    theta = []
    values = []
    missing = []

    for _, radar_row in radar_rows.iterrows():
        metric_name = str(radar_row.get("metric_name", "")).strip()
        label = str(radar_row.get("radar_label", radar_row.get("metric_label", metric_name))).strip()
        direction = str(radar_row.get("direction", "higher")).strip().lower()

        if metric_name not in TOTAL_SEASON_DF.columns:
            missing.append(metric_name)
            continue

        score = contextual_metric_score(
            TOTAL_SEASON_DF,
            wyscout_row,
            metric_name,
            direction,
            str(radar_row.get("position_group", "")),
            radar_row.get("reference_low", np.nan),
            radar_row.get("reference_high", np.nan),
        )
        if score is None:
            missing.append(metric_name)
            continue

        theta.append(label)
        values.append(score)

    if not values:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No radar metrics available for this role", showarrow=False, font=dict(color=DBU_MUTED))],
            margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=display_role_name(role_result["role_name"]),
            line=dict(width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=25, b=20),
        height=260,
        showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8), gridcolor="rgba(255,255,255,.12)"),
            angularaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,.10)"),
        ),
    )
    return fig


def role_radar_component(row: pd.Series):
    role_result = get_best_role_result(row)
    return html.Div(
        [
            html.Div("Role radar", style={**S["label"], "margin": "0 0 4px"}),
            dcc.Graph(
                figure=build_role_radar_figure(row, role_result),
                config={"displayModeBar": False},
                style={"height": "270px"},
            ),
        ],
        style={
            "minWidth": "280px",
            "maxWidth": "330px",
            "background": "rgba(0,0,0,.16)",
            "border": f"1px solid {DBU_BORDER}",
            "borderRadius": "18px",
            "padding": "12px",
        },
    )



# ============================================================
# SIMILAR PLAYERS — Wyscout Total Season only
# ============================================================
def normalize_metric_value(raw_value, reference_low, reference_high, direction="higher"):
    """Normalize a raw Wyscout value to 0–1 using Excel references.

    This is the same v11 logic used by role-fit/radar: no percentiles.
    """
    try:
        raw_value = float(raw_value)
        reference_low = float(reference_low)
        reference_high = float(reference_high)
    except Exception:
        return None

    if pd.isna(raw_value) or pd.isna(reference_low) or pd.isna(reference_high):
        return None

    if reference_high == reference_low:
        return None

    score01 = (raw_value - reference_low) / (reference_high - reference_low)
    score01 = max(0.0, min(1.0, score01))

    if str(direction).strip().lower() == "lower":
        score01 = 1.0 - score01

    return float(score01)


def get_role_metric_config(role_id: str) -> pd.DataFrame:
    """Metric configuration for one role_id, based only on ROLE_METRIC_TEMPLATES."""
    if ROLE_TEMPLATES.empty:
        return pd.DataFrame()

    cfg = ROLE_TEMPLATES.copy()
    cfg = cfg[
        (cfg["role_id"].astype(str) == str(role_id))
        & (cfg["score_use"])
        & (cfg["metric_available"])
        & (cfg["weight"] > 0)
        & (~cfg["metric_name"].astype(str).str.lower().str.contains("height", na=False))
    ].copy()

    if cfg.empty:
        return cfg

    cfg = cfg.sort_values("weight", ascending=False)
    return cfg.drop_duplicates("metric_name")


def get_eligible_roles_silent(row: pd.Series) -> pd.DataFrame:
    """Same purpose as get_eligible_roles, but without debug prints for similarity loops."""
    normalized_position = get_normalized_position(row)
    position_col = get_position_map_column()
    role_col = get_role_id_column()

    if not normalized_position or ROLE_MAP.empty or position_col is None or role_col is None:
        return pd.DataFrame(columns=["role_id", "role_name", "position_group"])

    map_positions = ROLE_MAP[position_col].apply(normalize_position_code)
    mapping_rows = ROLE_MAP[map_positions == normalized_position].copy()

    if mapping_rows.empty:
        return pd.DataFrame(columns=["role_id", "role_name", "position_group"])

    role_rows = []
    for _, map_row in mapping_rows.iterrows():
        for mapped_role_id in split_role_ids(map_row.get(role_col)):
            role_name = ""
            if "role_name" in map_row.index and map_row.get("role_name"):
                role_name = str(map_row.get("role_name")).strip()
            if not role_name:
                role_names = ROLE_TEMPLATES[ROLE_TEMPLATES["role_id"] == mapped_role_id]["role_name"].dropna().unique()
                role_name = str(role_names[0]) if len(role_names) else mapped_role_id

            role_rows.append(
                {
                    "role_id": mapped_role_id,
                    "role_name": role_name,
                    "position_group": str(map_row.get("position_group", "")).strip(),
                }
            )

    return pd.DataFrame(role_rows).drop_duplicates("role_id") if role_rows else pd.DataFrame(columns=["role_id", "role_name", "position_group"])


def get_position_group_metric_config(row: pd.Series) -> pd.DataFrame:
    """Broad metric set from all roles compatible with the player's mapped position."""
    eligible = get_eligible_roles_silent(row)

    if eligible.empty:
        return pd.DataFrame()

    role_ids = eligible["role_id"].astype(str).unique().tolist()

    cfg = ROLE_TEMPLATES.copy()
    cfg = cfg[
        (cfg["role_id"].astype(str).isin(role_ids))
        & (cfg["score_use"])
        & (cfg["metric_available"])
        & (cfg["weight"] > 0)
        & (~cfg["metric_name"].astype(str).str.lower().str.contains("height", na=False))
    ].copy()

    if cfg.empty:
        return cfg

    cfg = cfg.sort_values("weight", ascending=False)
    return cfg.drop_duplicates("metric_name")


def build_player_vector(player_row: pd.Series, metrics_config: pd.DataFrame):
    """Build a normalized tactical-profile vector for one player.

    Returns:
    - values: {metric_name: normalized 0–1 score}
    - weights: {metric_name: metric weight}
    """
    values = {}
    weights = {}

    if metrics_config is None or metrics_config.empty:
        return values, weights

    for _, metric in metrics_config.iterrows():
        metric_name = metric.get("metric_name")

        if metric_name not in player_row.index:
            continue

        score01 = normalize_metric_value(
            player_row.get(metric_name),
            metric.get("reference_low", np.nan),
            metric.get("reference_high", np.nan),
            metric.get("direction", "higher"),
        )

        if score01 is None:
            continue

        values[metric_name] = score01
        weights[metric_name] = float(metric.get("weight", 1) or 1)

    return values, weights


def calculate_cosine_similarity(vector_a: dict, vector_b: dict):
    common_metrics = [metric for metric in vector_a.keys() if metric in vector_b]

    if not common_metrics:
        return None, []

    a = np.array([vector_a[metric] for metric in common_metrics], dtype=float)
    b = np.array([vector_b[metric] for metric in common_metrics], dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return None, common_metrics

    return float(np.dot(a, b) / denominator), common_metrics


def calculate_level_score(candidate_vector: dict, weights: dict, common_metrics: list[str]):
    """Weighted average of candidate's normalized scores in the common metric set."""
    if not common_metrics:
        return None

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name in common_metrics:
        if metric_name not in candidate_vector:
            continue

        weight = float(weights.get(metric_name, 1) or 1)
        weighted_sum += candidate_vector[metric_name] * 100 * weight
        total_weight += weight

    if total_weight <= 0:
        return None

    return weighted_sum / total_weight


def simple_role_score_for_row(player_row: pd.Series, role_id: str):
    """Role score for a candidate using the current v11 raw-reference role logic."""
    cfg = get_role_metric_config(role_id)

    if cfg.empty:
        return None

    player_vector, player_weights = build_player_vector(player_row, cfg)

    if not player_vector:
        return None

    return calculate_level_score(player_vector, player_weights, list(player_vector.keys()))


def simple_best_role_for_row_silent(player_row: pd.Series) -> Optional[dict]:
    """Best role for a candidate without using the verbose debug role-fit function."""
    eligible = get_eligible_roles_silent(player_row)

    if eligible.empty:
        return None

    best_result = None

    for _, role in eligible.iterrows():
        role_id = str(role.get("role_id", "")).strip()
        role_name = str(role.get("role_name", role_id)).strip()
        score = simple_role_score_for_row(player_row, role_id)

        if score is None:
            continue

        result = {
            "role_id": role_id,
            "role_name": display_role_name(role_name),
            "role_fit_score": float(score),
        }

        if best_result is None or result["role_fit_score"] > best_result["role_fit_score"]:
            best_result = result

    return best_result


def diversify_similar_players(
    results_df: pd.DataFrame,
    top_n: int = 3,
    max_per_team: int = 1,
    max_per_league: int = 2,
    prefer_role_variety: bool = True,
) -> pd.DataFrame:
    """Return a more varied recommendation list.

    The raw similarity score is still respected, but the final visible list avoids
    showing three players from the same club/league/profile whenever good
    alternatives exist.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    ranked = results_df.sort_values("final_similarity_score", ascending=False).copy()

    selected_rows = []
    team_counts = {}
    league_counts = {}
    used_roles = set()

    for _, row in ranked.iterrows():
        team = str(row.get("team", ""))
        league = str(row.get("league", ""))
        role = str(row.get("best_role", ""))

        if team_counts.get(team, 0) >= max_per_team:
            continue
        if league_counts.get(league, 0) >= max_per_league:
            continue
        if prefer_role_variety and role in used_roles and len(used_roles) < top_n:
            continue

        selected_rows.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        league_counts[league] = league_counts.get(league, 0) + 1
        if role:
            used_roles.add(role)

        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) < top_n:
        selected_keys = {str(r.get("player_key", "")) for r in selected_rows}
        for _, row in ranked.iterrows():
            key = str(row.get("player_key", ""))
            if key in selected_keys:
                continue

            team = str(row.get("team", ""))
            league = str(row.get("league", ""))

            if team_counts.get(team, 0) >= max_per_team:
                continue
            if league_counts.get(league, 0) >= max_per_league:
                continue

            selected_rows.append(row)
            selected_keys.add(key)
            team_counts[team] = team_counts.get(team, 0) + 1
            league_counts[league] = league_counts.get(league, 0) + 1

            if len(selected_rows) >= top_n:
                break

    if len(selected_rows) < top_n:
        selected_keys = {str(r.get("player_key", "")) for r in selected_rows}
        for _, row in ranked.iterrows():
            key = str(row.get("player_key", ""))
            if key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(key)
            if len(selected_rows) >= top_n:
                break

    if not selected_rows:
        return ranked.head(top_n)

    return pd.DataFrame(selected_rows).head(top_n).reset_index(drop=True)


def choose_similarity_role(reference_row: pd.Series, role_id: Optional[str] = None):
    """Pick the role used for similarity search."""
    if role_id:
        role_names = ROLE_TEMPLATES[ROLE_TEMPLATES["role_id"].astype(str) == str(role_id)]["role_name"].dropna().unique()
        role_name = str(role_names[0]) if len(role_names) else str(role_id)
        return {"role_id": str(role_id), "role_name": role_name}

    role_scores = calculate_role_fit_scores(TOTAL_SEASON_DF, reference_row)

    if role_scores:
        return role_scores[0]

    eligible = get_eligible_roles_silent(reference_row)

    if eligible.empty:
        return None

    first = eligible.iloc[0]
    return {
        "role_id": str(first.get("role_id", "")),
        "role_name": str(first.get("role_name", first.get("role_id", ""))),
    }


def same_player(a: pd.Series, b: pd.Series) -> bool:
    if "player_key" in a.index and "player_key" in b.index:
        return str(a.get("player_key")) == str(b.get("player_key"))

    if "player_id" in a.index and "player_id" in b.index:
        return str(a.get("player_id")) == str(b.get("player_id"))

    return (
        str(a.get("player_name", "")).strip().lower()
        == str(b.get("player_name", "")).strip().lower()
        and str(a.get("team_name_unified", "")).strip().lower()
        == str(b.get("team_name_unified", "")).strip().lower()
    )


def apply_similarity_filters(candidates: pd.DataFrame, filters: Optional[dict] = None) -> pd.DataFrame:
    if not filters:
        return candidates

    out = candidates.copy()

    age_max = filters.get("age_max")
    if age_max is not None and "age" in out.columns:
        out = out[pd.to_numeric(out["age"], errors="coerce").fillna(999) <= age_max]

    leagues = filters.get("leagues")
    if leagues and "league_unified" in out.columns:
        out = out[out["league_unified"].isin(leagues)]

    nationalities = filters.get("nationalities")
    if nationalities and "nationality" in out.columns:
        out = out[out["nationality"].isin(nationalities)]

    teams = filters.get("teams")
    if teams and "team_name_unified" in out.columns:
        out = out[out["team_name_unified"].isin(teams)]

    return out


def get_similar_players(
    reference_player_id=None,
    total_season_df: Optional[pd.DataFrame] = None,
    role_metric_templates_df: Optional[pd.DataFrame] = None,
    radar_reference_df: Optional[pd.DataFrame] = None,
    position_role_map_df: Optional[pd.DataFrame] = None,
    roles_df: Optional[pd.DataFrame] = None,
    role_scores_df=None,
    role_id: Optional[str] = None,
    mode: str = "same_role",
    min_minutes: int = 700,
    top_n: int = 20,
    filters: Optional[dict] = None,
    reference_row: Optional[pd.Series] = None,
    min_common_metrics: int = 6,
) -> pd.DataFrame:
    """Recommend similar players using Wyscout Total Season and v11 raw references.

    final_similarity_score =
        0.50 * profile_similarity
      + 0.35 * level_score
      + 0.15 * role_fit_score
    """
    df = total_season_df if total_season_df is not None else TOTAL_SEASON_DF

    if df is None or df.empty:
        return pd.DataFrame()

    if reference_row is None:
        if reference_player_id is None:
            return pd.DataFrame()

        if "player_key" in df.columns:
            ref_df = df[df["player_key"].astype(str) == str(reference_player_id)]
        elif "player_id" in df.columns:
            ref_df = df[df["player_id"].astype(str) == str(reference_player_id)]
        else:
            ref_df = pd.DataFrame()

        if ref_df.empty:
            return pd.DataFrame()

        reference_row = ref_df.iloc[0]
    else:
        reference_row = get_wyscout_player_row(reference_row)

    if reference_row is None:
        return pd.DataFrame()

    selected_role = choose_similarity_role(reference_row, role_id=role_id)

    if not selected_role:
        return pd.DataFrame()

    selected_role_id = selected_role["role_id"]
    selected_role_name = selected_role["role_name"]

    if mode == "same_role":
        metrics_config = get_role_metric_config(selected_role_id)
    else:
        metrics_config = get_position_group_metric_config(reference_row)

    if metrics_config is None or metrics_config.empty:
        return pd.DataFrame()

    reference_vector, reference_weights = build_player_vector(reference_row, metrics_config)

    if not reference_vector:
        return pd.DataFrame()

    reference_eligible_roles = set(get_eligible_roles_silent(reference_row)["role_id"].astype(str).tolist())

    candidates = df.copy()

    if "minutes_played" in candidates.columns:
        minutes = pd.to_numeric(candidates["minutes_played"], errors="coerce").fillna(0)
        candidates = candidates[minutes >= min_minutes]

    candidates = apply_similarity_filters(candidates, filters)

    results = []

    for _, candidate in candidates.iterrows():
        if same_player(reference_row, candidate):
            continue

        candidate_eligible_roles = set(get_eligible_roles_silent(candidate)["role_id"].astype(str).tolist())

        if mode == "same_role":
            if selected_role_id not in candidate_eligible_roles:
                continue
        else:
            if not reference_eligible_roles.intersection(candidate_eligible_roles):
                continue

        candidate_vector, candidate_weights = build_player_vector(candidate, metrics_config)
        cosine_value, common_metrics = calculate_cosine_similarity(reference_vector, candidate_vector)

        if cosine_value is None:
            continue

        if len(common_metrics) < min_common_metrics:
            continue

        profile_similarity = cosine_value * 100
        level_score = calculate_level_score(candidate_vector, candidate_weights, common_metrics)

        if level_score is None:
            continue

        candidate_best_role = simple_best_role_for_row_silent(candidate)
        candidate_best_role_name = (
            candidate_best_role.get("role_name")
            if candidate_best_role is not None
            else selected_role_name
        )

        if mode == "same_role":
            role_fit_score = simple_role_score_for_row(candidate, selected_role_id)
        else:
            role_fit_score = (
                candidate_best_role.get("role_fit_score")
                if candidate_best_role is not None
                else None
            )

        if role_fit_score is None:
            role_fit_score = level_score

        final_similarity_score = (
            0.50 * profile_similarity
            + 0.35 * level_score
            + 0.15 * role_fit_score
        )

        positive_traits = []
        for metric_name in common_metrics:
            label_series = metrics_config.loc[metrics_config["metric_name"] == metric_name, "metric_label"]
            label = str(label_series.iloc[0]) if not label_series.empty else metric_name
            diff = abs(reference_vector[metric_name] - candidate_vector[metric_name])
            positive_traits.append((label, diff))

        positive_traits = [
            label for label, _ in sorted(positive_traits, key=lambda x: x[1])[:3]
        ]

        results.append(
            {
                "player_key": candidate.get("player_key", candidate.get("player_id", "")),
                "player_name": candidate.get("player_name", "Unknown"),
                "team": candidate.get("team_name_unified", "-"),
                "league": candidate.get("league_unified", "-"),
                "position": candidate.get("position_role_detailed", candidate.get("position_role_unified", "-")),
                "age": candidate.get("age", np.nan),
                "minutes": candidate.get("minutes_played", np.nan),
                "best_role": display_role_name(candidate_best_role_name),
                "final_similarity_score": final_similarity_score,
                "profile_similarity": profile_similarity,
                "level_score": level_score,
                "role_fit_score": role_fit_score,
                "common_metrics": len(common_metrics),
                "top_matching_traits": ", ".join(positive_traits),
            }
        )

    output = pd.DataFrame(results)

    if output.empty:
        return output

    return diversify_similar_players(
        output,
        top_n=top_n,
        max_per_team=1,
        max_per_league=2,
        prefer_role_variety=(mode != "same_role"),
    )


def similar_players_panel(row: pd.Series):
    sims = get_similar_players(
        reference_row=row,
        mode="same_position_group",
        min_minutes=700,
        top_n=3,
        min_common_metrics=6,
    )

    base_style = {
        "background": "rgba(0,0,0,.16)",
        "border": f"1px solid {DBU_BORDER}",
        "borderRadius": "18px",
        "padding": "12px",
        "height": "100%",
    }

    if sims.empty:
        return html.Div(
            [
                html.Div("Similar players", style={**S["label"], "margin": "0 0 8px"}),
                html.Div(
                    "No similar Wyscout players found.",
                    style={"color": DBU_MUTED, "fontSize": "12px"},
                ),
            ],
            style=base_style,
        )

    return html.Div(
        [
            html.Div("Similar players", style={**S["label"], "margin": "0 0 8px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                sim["player_name"],
                                style={"fontWeight": "900", "fontSize": "14px", "lineHeight": "1.1"},
                            ),
                            html.Div(
                                f"{sim['position']} · {sim['team']}",
                                style={"color": DBU_MUTED, "fontSize": "11px", "marginTop": "4px"},
                            ),
                            html.Div(
                                f"Similarity {sim['final_similarity_score']:.1f}",
                                style={"fontSize": "12px", "fontWeight": "900", "marginTop": "8px"},
                            ),
                            html.Div(
                                f"Profile {sim['profile_similarity']:.1f} · Level {sim['level_score']:.1f} · Role {sim['role_fit_score']:.1f}",
                                style={"fontSize": "10px", "color": DBU_MUTED, "marginTop": "3px"},
                            ),
                            html.Div(
                                f"{int(sim['common_metrics'])} shared metrics",
                                style={"fontSize": "10px", "color": DBU_MUTED, "marginTop": "3px"},
                            ),
                        ],
                        id={"type": "player-card", "player_id": str(sim.get("player_key", ""))},
                        n_clicks=0,
                        title="Open player profile",
                        style={
                            "background": "linear-gradient(180deg, rgba(255,255,255,.045), rgba(212,0,0,.055))",
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "14px",
                            "padding": "11px",
                            "cursor": "pointer",
                            "minHeight": "112px",
                            "transition": "all .16s ease",
                        },
                    )
                    for _, sim in sims.iterrows()
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, minmax(160px, 1fr))",
                    "gap": "10px",
                    "alignItems": "stretch",
                },
            ),
        ],
        style=base_style,
    )

def debug_role_fit_for_player(
    player_name: str,
    player_row: pd.Series,
    total_season_df: pd.DataFrame,
    role_templates: pd.DataFrame,
    position_role_map: pd.DataFrame,
    normalized_position: str,
    top_n_metrics: int = 25,
):
    """Temporary debug helper for v10 role-fit validation. Uses raw references, not percentiles."""

    def to_bool(x):
        return str(x).strip().upper() in {"TRUE", "1", "YES", "Y"}

    print()
    print("==============================")
    print(f"DEBUG ROLE FIT: {player_name}")
    print("==============================")
    print("Normalized position:", normalized_position)

    position_col = get_position_map_column() or "position_code"
    role_col = get_role_id_column() or "role_id"

    role_map = position_role_map.copy()
    if position_col not in role_map.columns:
        print(f"ERROR: POSITION_ROLE_MAP missing position column: {position_col}")
        return None

    role_map[position_col] = role_map[position_col].apply(normalize_position_code)
    eligible = role_map[role_map[position_col] == normalize_position_code(normalized_position)].copy()

    print()
    print("Eligible role map:")
    cols_to_print = [c for c in [position_col, "position_group", role_col, "role_name"] if c in eligible.columns]
    print(eligible[cols_to_print] if not eligible.empty else "EMPTY")

    if eligible.empty:
        print()
        print("ERROR: No eligible roles for this position.")
        return None

    eligible_role_ids = []
    for value in eligible[role_col].dropna().tolist():
        eligible_role_ids.extend(split_role_ids(value))
    eligible_role_ids = list(dict.fromkeys(eligible_role_ids))
    print("Eligible role ids:", eligible_role_ids)

    rt = role_templates.copy()
    rt["score_use_bool"] = rt["score_use"].apply(to_bool)
    rt = rt[(rt["role_id"].isin(eligible_role_ids)) & (rt["score_use_bool"])].copy()

    print()
    print("Template rows found:", len(rt))
    if rt.empty:
        print("ERROR: No role template rows found.")
        return None

    available_cols = set(total_season_df.columns)
    rt["metric_available"] = rt["metric_name"].isin(available_cols)
    missing = rt.loc[~rt["metric_available"], "metric_name"].dropna().unique().tolist()

    print()
    print("Missing metric columns:", len(missing))
    print(missing[:40])

    rt = rt[rt["metric_available"]].copy()
    if rt.empty:
        print("ERROR: No matching metric columns found.")
        return None

    all_results = []
    role_scores = []

    for role_id, group in rt.groupby("role_id"):
        role_name = group["role_name"].iloc[0]
        weighted_sum = 0.0
        total_weight = 0.0
        rows = []

        for _, m in group.iterrows():
            col = m["metric_name"]
            label = m.get("metric_label", col)
            weight = float(m.get("weight", 0) or 0)
            direction = str(m.get("direction", "higher")).strip().lower()
            competence = m.get("competence", "")
            signal_type = m.get("signal_type", "")
            reference_low = m.get("reference_low", np.nan)
            reference_high = m.get("reference_high", np.nan)

            raw_value = player_row.get(col, np.nan)
            try:
                raw_value = float(raw_value)
            except Exception:
                continue

            if pd.isna(raw_value):
                continue

            metric_score = raw_reference_metric_score(raw_value, reference_low, reference_high, direction)
            if metric_score is None:
                continue

            weighted_score = metric_score * weight
            weighted_sum += weighted_score
            total_weight += weight

            rows.append(
                {
                    "player": player_name,
                    "role_id": role_id,
                    "role_name": role_name,
                    "competence": competence,
                    "signal_type": signal_type,
                    "metric_name": col,
                    "metric_label": label,
                    "raw_value": raw_value,
                    "reference_low": reference_low,
                    "reference_high": reference_high,
                    "direction": direction,
                    "metric_score": round(metric_score, 1),
                    "weight": weight,
                    "weighted_score": round(weighted_score, 2),
                }
            )

        role_score = weighted_sum / total_weight if total_weight > 0 else np.nan
        role_scores.append(
            {
                "role_id": role_id,
                "role_name": role_name,
                "role_fit_score": round(role_score, 1) if pd.notna(role_score) else np.nan,
                "total_weight_used": round(total_weight, 3),
                "metrics_used": len(rows),
            }
        )
        all_results.extend(rows)

    role_scores_df = pd.DataFrame(role_scores)
    if not role_scores_df.empty:
        role_scores_df = role_scores_df.sort_values("role_fit_score", ascending=False)

    contributions_df = pd.DataFrame(all_results)

    print()
    print("ROLE SCORES")
    print(role_scores_df)
    print()
    print("TOP METRIC CONTRIBUTIONS BY ROLE")

    if not contributions_df.empty:
        for role_name in role_scores_df["role_name"].tolist():
            print()
            print("------------------------------")
            print(role_name)
            print("------------------------------")
            role_contrib = contributions_df[contributions_df["role_name"] == role_name].sort_values(
                "weighted_score", ascending=False
            )
            print(
                role_contrib[
                    [
                        "metric_label",
                        "raw_value",
                        "reference_low",
                        "reference_high",
                        "direction",
                        "metric_score",
                        "weight",
                        "weighted_score",
                        "competence",
                        "signal_type",
                    ]
                ].head(top_n_metrics)
            )

    return role_scores_df, contributions_df


def debug_role_fit_for_player_name(player_name: str, top_n_metrics: int = 25):
    """Convenience wrapper: find a player by name and run role-fit debug."""
    if "player_name" not in TOTAL_SEASON_DF.columns:
        print("ERROR: player_name column not found in TOTAL_SEASON_DF")
        return None

    matches = TOTAL_SEASON_DF[
        TOTAL_SEASON_DF["player_name"].astype(str).str.contains(player_name, case=False, na=False)
    ]

    if matches.empty:
        print(f"ERROR: No Wyscout total-season row found for: {player_name}")
        return None

    player_row = matches.iloc[0]
    normalized_position = get_normalized_position(player_row)

    return debug_role_fit_for_player(
        player_name=player_row.get("player_name", player_name),
        player_row=player_row,
        total_season_df=TOTAL_SEASON_DF,
        role_templates=ROLE_TEMPLATES,
        position_role_map=ROLE_MAP,
        normalized_position=normalized_position,
        top_n_metrics=top_n_metrics,
    )


def debug_validation_players():
    debug_players = [
        "Vestergaard",
        "Andreas Christensen",
        "Joakim Mæhle",
        "Morten Hjulmand",
        "Rasmus Højlund",
        "Jonas Wind",
        "William Osula",
        "Christian Eriksen",
    ]

    results = {}
    for player_name in debug_players:
        print()
        print()
        print("################################")
        print(player_name)
        print("################################")
        results[player_name] = debug_role_fit_for_player_name(player_name)
    return results


def role_score_bar(score: float):
    score = 0 if pd.isna(score) else float(np.clip(score, 0, 100))
    color = percentile_color(score)
    return html.Div(
        [
            html.Div(
                style={
                    "width": f"{score:.0f}%",
                    "height": "100%",
                    "borderRadius": "999px",
                    "background": color,
                }
            )
        ],
        style={
            "height": "8px",
            "background": "rgba(255,255,255,.08)",
            "borderRadius": "999px",
            "overflow": "hidden",
            "marginTop": "8px",
        },
    )


def display_role_name(role_name: str) -> str:
    """UI-only role naming override.

    v10 may keep internal DMF role_ids as dmf_*_recycler, but the visible label
    should read non progressor instead of Recycler.
    """
    text = str(role_name)
    text = text.replace("Recycler", "non progressor")
    text = text.replace("recycler", "non progressor")
    return text


def role_modifier_pills(modifiers: list[dict]):
    if not modifiers:
        return html.Div("No strong modifiers detected yet.", style={"color": DBU_MUTED, "fontSize": "12px"})

    return html.Div(
        [
            html.Span(
                str(m["tag"]),
                style={
                    **S["pill"],
                    "background": "rgba(212,0,0,.22)",
                    "border": f"1px solid {DBU_RED_SOFT}",
                    "marginBottom": "6px",
                },
            )
            for m in modifiers
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
    )


def deduplicate_metrics_for_display(top_metrics: list[dict]) -> list[dict]:
    """Deduplicate only the visual Main Role Signals list.

    A metric can legitimately appear multiple times in the role template because it
    contributes to different competences/modifiers. Keep that for scoring, but
    show each real Wyscout column only once in the UI.
    """
    seen_metrics = set()
    unique_metrics = []

    for metric in top_metrics or []:
        metric_name = metric.get("metric_name")
        if not metric_name:
            continue
        if metric_name in seen_metrics:
            continue

        seen_metrics.add(metric_name)
        unique_metrics.append(metric)

    return unique_metrics


def format_role_signal_value(row: pd.Series, metric: dict) -> str:
    """Display only the real Wyscout value for Main Role Signals.

    v11 keeps internal raw-reference scores for role selection/debug, but the UI
    should not show the internal S score next to the metric value.
    """
    metric_name = metric.get("metric_name")
    signal_type = str(metric.get("signal_type", "")).lower()
    metric_label = str(metric.get("metric_label", "")).lower()

    wyscout_row = get_wyscout_player_row(row)
    value = metric.get("raw_value", np.nan)

    if wyscout_row is not None and metric_name in wyscout_row.index:
        value = wyscout_row.get(metric_name, value)

    if pd.isna(value):
        return "—"

    if "percentage" in signal_type or "%" in metric_label or "%" in str(metric_name):
        return format_percentage(value)

    return format_value(value)


def role_fit_panel(row: pd.Series):
    role_scores = calculate_role_fit_scores(TOTAL_SEASON_DF, row)

    if not role_scores:
        return html.Div(
            [
                html.Div("Role fit", style={**S["label"], "margin": "0 0 8px"}),
                html.Div(
                    "No eligible Wyscout role template metrics available for this player yet.",
                    style={"color": DBU_MUTED, "fontSize": "13px"},
                ),
            ],
            style={
                "background": "rgba(255,255,255,.025)",
                "border": f"1px solid {DBU_BORDER}",
                "borderRadius": "18px",
                "padding": "14px",
                "marginTop": "14px",
            },
        )

    best = role_scores[0]
    secondary = role_scores[1] if len(role_scores) > 1 else None
    top_metrics = sorted(
        best.get("used_metrics", []),
        key=lambda x: x.get("weighted_score", x.get("score", 0) * x.get("weight", 0)),
        reverse=True,
    )
    top_metrics = deduplicate_metrics_for_display(top_metrics)[:6]

    return html.Div(
        [
            html.Div("Role fit", style={**S["label"], "margin": "0 0 10px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Best role", style={"color": DBU_MUTED, "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": ".1em"}),
                            html.H3(display_role_name(best["role_name"]), style={"margin": "4px 0 0", "fontSize": "24px"}),
                        ],
                        style={
                            "background": "linear-gradient(135deg, rgba(212,0,0,.16), rgba(255,255,255,.035))",
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "16px",
                            "padding": "14px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div("Secondary role", style={"color": DBU_MUTED, "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": ".1em"}),
                            html.H4(display_role_name(secondary["role_name"]) if secondary else "—", style={"margin": "5px 0", "fontSize": "19px"}),
                            html.Div("Modifiers", style={**S["label"], "margin": "14px 0 8px"}),
                            role_modifier_pills(best.get("modifiers", [])),
                        ],
                        style={
                            "background": "rgba(255,255,255,.035)",
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "16px",
                            "padding": "14px",
                        },
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "minmax(220px, .9fr) minmax(240px, 1.1fr)", "gap": "12px"},
            ),
            html.Div(
                [
                    html.Div("Main role signals", style={**S["label"], "margin": "14px 0 8px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(metric["metric_label"], style={"fontSize": "13px", "fontWeight": "800"}),
                                    html.Div(format_role_signal_value(row, metric), style={"fontSize": "15px", "fontWeight": "900", "color": DBU_TEXT}),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "gap": "12px",
                                    "padding": "7px 0",
                                    "borderBottom": "1px solid rgba(255,255,255,.06)",
                                },
                            )
                            for metric in top_metrics
                        ],
                    ),
                ]
            ),
        ],
        style={
            "background": "rgba(0,0,0,.16)",
            "border": "1px solid rgba(255,255,255,.06)",
            "borderRadius": "20px",
            "padding": "14px",
            "marginTop": "14px",
            "boxShadow": "0 8px 18px rgba(0,0,0,.16)",
        },
    )


def profile_toggle_button(label: str, value: str, active_value: str, component_id: str):
    active = value == active_value
    return html.Button(
        label,
        id={"type": component_id, "value": value},
        n_clicks=0,
        style={
            "background": DBU_RED if active else "rgba(255,255,255,.05)",
            "color": DBU_TEXT,
            "border": f"1px solid {DBU_RED_SOFT}" if active else f"1px solid {DBU_BORDER}",
            "borderRadius": "999px",
            "padding": "6px 11px",
            "fontSize": "12px",
            "fontWeight": "800",
            "cursor": "pointer",
            "boxShadow": "0 0 14px rgba(212,0,0,.20)" if active else "none",
        },
    )


def player_profile(row: Optional[pd.Series], mode: str, metric_mode: str):
    if row is None:
        return html.Div(
            [
                html.Div("Player profile", style={**S["label"], "marginTop": "0"}),
                html.H2("Select a player card to view profile", style={"margin": 0}),
                html.P("Click any player card below to open the scouting profile.", style=S["subtitle"]),
            ]
        )

    metrics_block = (
        league_profile_metrics(row, metric_mode or "per90")
        if mode == "league"
        else wyscout_profile_metrics(row, metric_mode or "per90")
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Button(
                        "← Back to player search",
                        id="back_to_player_search_btn",
                        n_clicks=0,
                        style={
                            "background": "rgba(255,255,255,.055)",
                            "color": DBU_TEXT,
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "999px",
                            "padding": "8px 13px",
                            "fontWeight": "850",
                            "fontSize": "12px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="add_shadow_slot_dropdown",
                                options=[
                                    {"label": opt["label"], "value": opt["value"]}
                                    for opt in SHADOW_SLOT_OPTIONS
                                ],
                                value=None,
                                clearable=False,
                                searchable=False,
                                placeholder="+ Add to shadow squad",
                                className="shadow-add-dropdown",
                                style={
                                    "width": "230px",
                                    "fontSize": "12px",
                                    "fontWeight": "900",
                                },
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "10px", "marginBottom": "14px", "flexWrap": "wrap"},
            ),
            player_bio_card(row),
            role_fit_panel(row),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Profile scope", style={**S["label"], "margin": "0 0 8px"}),
                            html.Div(
                                [
                                    profile_toggle_button("League only", "league", mode or "league", "profile-scope-btn"),
                                    profile_toggle_button("Total season", "total", mode or "league", "profile-scope-btn"),
                                ],
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Div(
                                "League only metric view" if (mode or "league") == "league" else "Total season metric view",
                                style={**S["label"], "margin": "14px 0 8px"},
                            ),
                            html.Div(
                                [
                                    profile_toggle_button("Per 90", "per90", metric_mode or "per90", "metric-view-btn"),
                                    profile_toggle_button("Total", "total", metric_mode or "per90", "metric-view-btn"),
                                ],
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                            ),
                            html.Div(
                                "Per 90 is the default normalized scouting view. Total shows volume/accumulation.",
                                style={"color": DBU_MUTED, "fontSize": "11px", "marginTop": "7px"},
                            ),
                        ],
                        style={
                            "marginTop": "10px",
                            "paddingTop": "10px",
                            "borderTop": f"1px solid {DBU_BORDER}",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gap": "4px",
                    "marginTop": "14px",
                    "padding": "12px",
                    "background": "rgba(255,255,255,.025)",
                    "border": f"1px solid {DBU_BORDER}",
                    "borderRadius": "16px",
                    "maxWidth": "520px",
                },
            ),
            metrics_block,
        ]
    )



# ============================================================
# SHADOW SQUAD / 4-2-3-1 BUILDER
# ============================================================

SHADOW_FORMATION_SLOTS = [
    "CF",
    "LW",
    "AMF",
    "RW",
    "DMF-L",
    "DMF-R",
    "LB",
    "LCB",
    "RCB",
    "RB",
    "GK",
    "BENCH",
]

SHADOW_SLOT_LABELS = {
    "CF": "CF",
    "LW": "LW",
    "AMF": "AMF",
    "RW": "RW",
    "DMF-L": "DMF L",
    "DMF-R": "DMF R",
    "LB": "LB",
    "LCB": "LCB",
    "RCB": "RCB",
    "RB": "RB",
    "GK": "GK",
    "BENCH": "Bench",
}

SHADOW_SLOT_OPTIONS = [
    {"label": SHADOW_SLOT_LABELS[slot], "value": slot}
    for slot in SHADOW_FORMATION_SLOTS
]

def get_player_row_by_key(player_key: str) -> Optional[pd.Series]:
    if player_key is None:
        return None
    match = DF[DF["player_key"].astype(str) == str(player_key)]
    if match.empty:
        return None
    return match.iloc[0]


def shadow_player_payload(row: pd.Series) -> dict:
    return {
        "player_key": str(row.get("player_key", row.get("player_id", ""))),
        "player_name": str(row.get("player_name", "Unknown")),
        "team": str(row.get("team_name_unified", "-")),
        "league": str(row.get("league_unified", "-")),
        "position": str(row.get("position_display", row.get("position_role_detailed", "-"))),
        "position_detail": str(row.get("position_role_detailed", "")),
        "position_role": str(row.get("position_role_unified", "")),
        "position_group": str(row.get("position_group_unified", "")),
        "age": row.get("age", np.nan),
        "minutes": row.get("minutes_played", np.nan),
        "assigned_slot": "BENCH",
    }


def infer_shadow_slots(player: dict) -> list[str]:
    code = normalize_position_code(player.get("position_detail") or player.get("position") or player.get("position_role"))
    group = str(player.get("position_group", ""))
    role = normalize_position_code(player.get("position_role", ""))

    if code == "GK" or role == "GK" or group == "Goalkeepers":
        return ["GK"]
    if code in {"LB", "LWB"}:
        return ["LB", "RB"]
    if code in {"RB", "RWB"}:
        return ["RB", "LB"]
    if code in {"LCB", "CB", "RCB"}:
        return ["LCB", "RCB"]
    if code in {"DMF", "LDMF", "RDMF"} or role in {"DMF", "DM"}:
        return ["DMF-L", "DMF-R"]
    if code in {"CMF", "LCMF", "RCMF"} or role in {"CMF", "CM"}:
        return ["DMF-L", "DMF-R", "AMF"]
    if code in {"AMF", "LAMF", "RAMF"} or role in {"AMF", "AM"}:
        return ["AMF", "LW", "RW"]
    if code in {"LW", "LWF", "LMF", "LF", "AML", "LWM"}:
        return ["LW", "RW", "AMF"]
    if code in {"RW", "RWF", "RMF", "RF", "AMR", "RWM"}:
        return ["RW", "LW", "AMF"]
    if code in {"CF", "ST", "STRIKER"} or role == "CF":
        return ["CF"]
    if group == "Defenders":
        return ["LCB", "RCB", "LB", "RB"]
    if group == "Midfielders":
        return ["DMF-L", "DMF-R", "AMF"]
    if group == "Forwards":
        return ["CF", "LW", "RW", "AMF"]
    return ["BENCH"]


def find_default_shadow_slot(player: dict, current_squad: list[dict]) -> str:
    """Suggest a first slot, but keep the user free to change it later."""
    occupied = {
        str(p.get("assigned_slot"))
        for p in current_squad or []
        if str(p.get("assigned_slot")) in SHADOW_FORMATION_SLOTS and str(p.get("assigned_slot")) != "BENCH"
    }

    for slot in infer_shadow_slots(player):
        if slot in SHADOW_FORMATION_SLOTS and slot != "BENCH" and slot not in occupied:
            return slot

    return "BENCH"


def assign_shadow_squad_slots(shadow_squad: list[dict]) -> tuple[dict, list[dict]]:
    slots = {slot: None for slot in ["GK", "LB", "LCB", "RCB", "RB", "DMF-L", "DMF-R", "LW", "AMF", "RW", "CF"]}
    bench = []

    for player in shadow_squad or []:
        assigned_slot = str(player.get("assigned_slot", "")).strip()

        # Backward compatibility for players already stored before manual slot assignment existed.
        if not assigned_slot or assigned_slot not in SHADOW_FORMATION_SLOTS:
            inferred = infer_shadow_slots(player)
            assigned_slot = inferred[0] if inferred else "BENCH"
            player["assigned_slot"] = assigned_slot

        if assigned_slot == "BENCH":
            bench.append(player)
            continue

        if assigned_slot in slots and slots[assigned_slot] is None:
            slots[assigned_slot] = player
        else:
            # If two players are assigned to the same formation slot, keep the first in the XI
            # and place the other in bench/depth until the coach chooses another slot.
            bench.append(player)

    return slots, bench


# ============================================================
# TEAM TACTICAL FIT LAYER FOR SHADOW SQUAD
# ============================================================
TEAM_STYLE_PROFILE = {
    "possession_dominance": 0.92,
    "short_passing_progression": 0.92,
    "central_penetration": 0.82,
    "high_pressing": 0.92,
    "opp_half_defensive_actions": 0.88,
    "territorial_attack": 0.85,
    "wide_attack": 0.78,
    "crossing_volume": 0.72,
    "box_presence": 0.80,
    "runs_in_behind_need": 0.85,
    "good_value_chances_need": 0.85,
    "second_ball_need": 0.78,
    "set_piece_defense_need": 0.75,
    "transition_attack": 0.72,
    "defensive_high_line": 0.80,
    "defensive_transition_risk": 0.65,
}

SHADOW_SLOT_REQUIREMENTS = {
    "GK": {
        "title": "Proactive GK",
        "priority": "Critical",
        "roles": ["Proactive GK", "Reactive GK"],
        "needs": ["Proactive sweeping", "Short build-up security", "Claim crosses"],
        "why": "This model prioritises a proactive goalkeeper because the team uses a high line, short build-up and aggressive pressing. Reactive GK is acceptable only as a fallback when shot-stopping and security are strong.",
    },
    "LB": {
        "title": "Powerhouse / Creative FB",
        "priority": "High",
        "roles": ["Powerhouse wingback", "Creative fullback"],
        "needs": ["Short-passing support", "Counterpressing", "Quality final action"],
        "why": "The team attacks wide, but needs quality and security, not just more crosses.",
    },
    "RB": {
        "title": "Powerhouse / Creative FB",
        "priority": "High",
        "roles": ["Powerhouse wingback", "Creative fullback"],
        "needs": ["Short-passing support", "Counterpressing", "Quality final action"],
        "why": "The team attacks wide, but needs quality and security, not just more crosses.",
    },
    "LCB": {
        "title": "Mobile/Physical + Initiative",
        "priority": "High",
        "roles": ["Mobile + initiative", "Physical + initiative", "Mobile + safe", "Physical + safe"],
        "needs": ["High-line defending", "Progress from deep", "Second-ball security"],
        "why": "A high defensive line needs centre backs who can anticipate and still progress play.",
    },
    "RCB": {
        "title": "Mobile/Physical + Initiative",
        "priority": "High",
        "roles": ["Mobile + initiative", "Physical + initiative", "Mobile + safe", "Physical + safe"],
        "needs": ["High-line defending", "Progress from deep", "Second-ball security"],
        "why": "A high defensive line needs centre backs who can anticipate and still progress play.",
    },
    "DMF-L": {
        "title": "Shielder/Recoverer + Progressor",
        "priority": "Very high",
        "roles": ["Shielder + progressor", "Recoverer + progressor", "Shielder + non progressor", "Recoverer + non progressor"],
        "needs": ["Protect losses", "Short progression", "Second balls"],
        "why": "The team dominates territory but needs protection behind attacks and progression through pressure.",
    },
    "DMF-R": {
        "title": "Shielder/Recoverer + Progressor",
        "priority": "Very high",
        "roles": ["Shielder + progressor", "Recoverer + progressor", "Shielder + non progressor", "Recoverer + non progressor"],
        "needs": ["Protect losses", "Short progression", "Second balls"],
        "why": "The team dominates territory but needs protection behind attacks and progression through pressure.",
    },
    "LW": {
        "title": "Interior creator / wide 10",
        "priority": "Very high",
        "roles": ["Extra midfielder", "Passing progressor", "Dribbling progressor", "Deep runner", "Wide dribbler"],
        "needs": ["Interior combinations", "xA / through-pass threat", "Arrive between lines"],
        "why": "For this model the left side can be used as an interior creator: a wide 10 who helps penetrate centrally, combine in short-passing sequences and create cleaner chances, not only a touchline dribbler.",
    },
    "RW": {
        "title": "Interior variety / efficient 1v1",
        "priority": "Very high",
        "roles": ["Extra midfielder", "Wide dribbler", "Deep runner"],
        "needs": ["Inside threat", "Runs in behind", "Counterpressing"],
        "why": "Wide volume already exists; the slot needs better decisions, inside threat and low-block solutions.",
    },
    "AMF": {
        "title": "Low-block solution",
        "priority": "Critical",
        "roles": ["Passing progressor", "Dribbling progressor", "Deep runner"],
        "needs": ["Central penetration", "Final pass", "Box arrival"],
        "why": "This is the main evolution slot: convert territory into cleaner chances against low blocks.",
    },
    "CF": {
        "title": "Box efficiency + pressure",
        "priority": "Very high",
        "roles": ["Deep runner", "Target man", "False nine", "Complete forward"],
        "needs": ["Runs in behind", "Good-value shots", "Press first line"],
        "why": "The team creates volume; the striker must turn territory into goals and add variety.",
    },
    "BENCH": {
        "title": "Depth / tactical alternative",
        "priority": "Flexible",
        "roles": ["Any useful specialist"],
        "needs": ["Cover roles", "Change game state", "Specific solution"],
        "why": "Bench players should offer either reliable depth or a tactical alternative.",
    },
}

TEAM_NEEDS = [
    {"label": "Short-passing progression", "why": "Sustain long passing sequences while still breaking lines through the middle."},
    {"label": "Runs in behind", "why": "Forwards and wide players must attack better receiving zones, not only circulate around the block."},
    {"label": "High pressing continuity", "why": "Keep PPDA low, recover high and avoid high-value shots against."},
    {"label": "Second balls + set-piece defence", "why": "Improve marking, duels and reactions after loose balls and defensive restarts."},
]


def fit_color(score: float) -> str:
    if pd.isna(score):
        return DBU_BORDER
    if score >= 72:
        return "#22c55e"
    if score >= 58:
        return "#eab308"
    return "#ef4444"


def fit_label(score: float) -> str:
    if pd.isna(score):
        return "Needs review"
    if score >= 72:
        return "Strong fit"
    if score >= 58:
        return "Usable fit"
    return "Weak fit"


# Selection-context adjustment: small league-strength layer for national-team use.
# This does NOT change the raw player archetype/role-fit logic. It only nudges the
# tactical/selection interpretation shown in the Shadow Squad cards.
LEAGUE_CONTEXT_ADJUSTMENTS = {
    "Premier League": 6,
    "LaLiga": 5,
    "La Liga": 5,
    "Bundesliga": 5,
    "Serie A": 5,
    "Ligue 1": 4,
    "Eredivisie": 2,
    "Belgian Pro League": 1,
    "Jupiler Pro League": 1,
    "Championship": 1,
    "MLS": 0,
    "Danish Superliga": 0,
    "Superliga": 0,
    "Liga Portugal": 2,
    "Primeira Liga": 2,
    "Austrian Bundesliga": -1,
    "Swiss Super League": -1,
    "Allsvenskan": -2,
    "Norwegian Eliteserien": -2,
}


def league_context_adjustment(row: pd.Series) -> tuple[int, str]:
    """Return a small league-context adjustment for selection use.

    The adjustment is intentionally capped and conservative. It rewards players
    producing in stronger contexts without making league strength more important
    than football role/tactical fit.
    """
    league = str(row.get("league_unified", row.get("league", ""))).strip()
    if not league or league.lower() in {"nan", "none", "-"}:
        return 0, "League context 0"

    # Exact match first.
    if league in LEAGUE_CONTEXT_ADJUSTMENTS:
        adj = LEAGUE_CONTEXT_ADJUSTMENTS[league]
    else:
        league_lower = league.lower()
        adj = 0
        for key, value in LEAGUE_CONTEXT_ADJUSTMENTS.items():
            if key.lower() in league_lower or league_lower in key.lower():
                adj = value
                break

    adj = int(max(-4, min(6, adj)))
    sign = "+" if adj > 0 else ""
    return adj, f"League context {sign}{adj}"


def apply_selection_context_adjustment(fit: dict, row: pd.Series) -> dict:
    """Apply league context to a fit dict without changing invalid-slot logic."""
    fit = dict(fit or {})

    if not fit.get("valid_position", True):
        fit["league_adjustment"] = 0
        fit["league_context"] = "No league adjustment"
        return fit

    score = fit.get("score", np.nan)
    adj, label = league_context_adjustment(row)
    fit["league_adjustment"] = adj
    fit["league_context"] = label

    if pd.notna(score):
        adjusted = float(np.clip(float(score) + adj, 0, 100))
        fit["raw_score_before_league"] = float(score)
        fit["score"] = adjusted
        # Recalculate label after league-context adjustment, keeping bench prefix if present.
        current_label = str(fit.get("label", ""))
        if " fit:" in current_label:
            prefix = current_label.split(":", 1)[0]
            fit["label"] = f"{prefix}: {fit_label(adjusted)}"
        elif current_label not in {"Invalid slot"}:
            fit["label"] = fit_label(adjusted)

    return fit


SLOT_POSITION_FAMILIES = {
    "GK": {"GK"},
    "LB": {"LB", "LWB"},
    "RB": {"RB", "RWB"},
    "LCB": {"CB", "LCB", "RCB"},
    "RCB": {"CB", "LCB", "RCB"},
    "DMF-L": {"DMF", "LDMF", "RDMF", "DM"},
    "DMF-R": {"DMF", "LDMF", "RDMF", "DM"},
    "LW": {"LW", "LWF", "LMF", "AML", "LWM", "LF", "WF"},
    "RW": {"RW", "RWF", "RMF", "AMR", "RWM", "RF", "WF"},
    "AMF": {"AMF", "LAMF", "RAMF", "AM", "CMF", "LCMF", "RCMF"},
    "CF": {"CF", "ST", "STRIKER", "RWF", "LWF"},
    "BENCH": set(),
}


def player_position_codes_for_slot_check(row: pd.Series) -> set[str]:
    codes = set()
    for col in ["position_role_detailed", "wyscout_Position_total", "position_display", "position_role_unified", "primary_position_clean"]:
        if col in row.index and pd.notna(row.get(col)):
            code = normalize_position_code(row.get(col))
            if code:
                codes.add(code)
    return codes


SLOT_POSITION_PENALTY_RULES = {
    "GK": {
        "ideal": {"GK"},
        "compatible": {},
        "stretch": {},
    },
    "LB": {
        "ideal": {"LB", "LWB"},
        "compatible": {"RB", "RWB", "LCB", "LMF", "LW"},
        "stretch": {"CB", "RCB", "DMF", "CMF", "LDMF", "RDMF"},
    },
    "RB": {
        "ideal": {"RB", "RWB"},
        "compatible": {"LB", "LWB", "RCB", "RMF", "RW"},
        "stretch": {"CB", "LCB", "DMF", "CMF", "LDMF", "RDMF"},
    },
    "LCB": {
        "ideal": {"CB", "LCB", "RCB"},
        "compatible": {"DMF", "LDMF", "RDMF", "LB", "LWB"},
        "stretch": {"RB", "RWB", "CMF", "LCMF", "RCMF"},
    },
    "RCB": {
        "ideal": {"CB", "LCB", "RCB"},
        "compatible": {"DMF", "LDMF", "RDMF", "RB", "RWB"},
        "stretch": {"LB", "LWB", "CMF", "LCMF", "RCMF"},
    },
    "DMF-L": {
        "ideal": {"DMF", "LDMF", "RDMF", "DM"},
        "compatible": {"CMF", "LCMF", "RCMF", "CB", "LCB", "RCB"},
        "stretch": {"AMF", "LAMF", "RAMF", "LB", "RB", "LWB", "RWB"},
    },
    "DMF-R": {
        "ideal": {"DMF", "LDMF", "RDMF", "DM"},
        "compatible": {"CMF", "LCMF", "RCMF", "CB", "LCB", "RCB"},
        "stretch": {"AMF", "LAMF", "RAMF", "LB", "RB", "LWB", "RWB"},
    },
    "LW": {
        "ideal": {"LW", "LWF", "LMF", "AML", "LWM", "LF", "WF"},
        "compatible": {"RW", "RWF", "RMF", "AMR", "RWM", "RF", "AMF", "LAMF", "RAMF"},
        "stretch": {"CF", "ST", "CMF", "LCMF", "RCMF", "LWB", "LB"},
    },
    "RW": {
        "ideal": {"RW", "RWF", "RMF", "AMR", "RWM", "RF", "WF"},
        "compatible": {"LW", "LWF", "LMF", "AML", "LWM", "LF", "AMF", "LAMF", "RAMF"},
        "stretch": {"CF", "ST", "CMF", "LCMF", "RCMF", "RWB", "RB"},
    },
    "AMF": {
        "ideal": {"AMF", "LAMF", "RAMF", "AM"},
        "compatible": {"CMF", "LCMF", "RCMF", "LW", "RW", "LWF", "RWF", "LMF", "RMF"},
        "stretch": {"CF", "ST", "DMF", "LDMF", "RDMF"},
    },
    "CF": {
        "ideal": {"CF", "ST", "STRIKER"},
        "compatible": {"RWF", "LWF", "RW", "LW", "AMF", "LAMF", "RAMF"},
        "stretch": {"LMF", "RMF", "CMF"},
    },
    "BENCH": {
        "ideal": set(),
        "compatible": set(),
        "stretch": set(),
    },
}


def slot_position_penalty(row: pd.Series, slot_key: str) -> dict:
    """Return position adaptation penalty instead of making every mismatch invalid.

    Only impossible GK/outfield swaps are hard invalid. Other changes are allowed
    but penalised according to how natural the adaptation is.
    """
    if slot_key == "BENCH":
        return {"valid": True, "penalty": 0, "level": "bench", "note": "Bench option"}

    player_codes = player_position_codes_for_slot_check(row)
    is_gk_player = "GK" in player_codes or str(row.get("position_role_unified", "")).strip().upper() == "GK"

    if slot_key == "GK" and not is_gk_player:
        return {"valid": False, "penalty": 100, "level": "invalid", "note": "Invalid: outfield player cannot play goalkeeper"}
    if slot_key != "GK" and is_gk_player:
        return {"valid": False, "penalty": 100, "level": "invalid", "note": "Invalid: goalkeeper cannot play outfield"}

    rules = SLOT_POSITION_PENALTY_RULES.get(slot_key, {})
    ideal = rules.get("ideal", set())
    compatible = rules.get("compatible", set())
    stretch = rules.get("stretch", set())

    if player_codes.intersection(ideal):
        return {"valid": True, "penalty": 0, "level": "natural", "note": "Natural position"}
    if player_codes.intersection(compatible):
        return {"valid": True, "penalty": 8, "level": "adapted", "note": "Adapted position: light penalty"}
    if player_codes.intersection(stretch):
        return {"valid": True, "penalty": 18, "level": "stretch", "note": "Stretch position: moderate penalty"}

    codes_text = ", ".join(sorted(player_codes)) if player_codes else "unknown"
    return {"valid": True, "penalty": 30, "level": "major_mismatch", "note": f"Major role change from {codes_text}"}


def fit_label_for_slot(score: float, is_position_valid: bool) -> str:
    if not is_position_valid:
        return "Invalid slot"
    return fit_label(score)


def normalize_role_for_match(role_name: str) -> str:
    return display_role_name(role_name).strip().lower().replace("-", " ")


def role_matches_slot(role_name: str, slot_key: str) -> bool:
    req = SHADOW_SLOT_REQUIREMENTS.get(slot_key, {})
    preferred = [normalize_role_for_match(r) for r in req.get("roles", [])]
    current = normalize_role_for_match(role_name)
    return any(p in current or current in p for p in preferred)


def tactical_slot_bonus(role_name: str, slot_key: str) -> tuple[int, str, list[str]]:
    """Team-style bonus based on the analyst description.

    The model is short-passing / possession based, presses high, can counter quickly,
    and needs better runs in behind, box reception, second balls and set-piece defence.
    """
    role = normalize_role_for_match(role_name)
    traits = []
    bonus = 0
    fit_type = "Role mismatch"

    if slot_key == "GK":
        if "proactive" in role:
            return 18, "Critical fit", ["High-line coverage", "Short build-up", "Sweeper behaviour"]
        if "reactive" in role:
            return 8, "Fallback GK", ["Shot-stopping fallback", "Needs build-up check", "Less proactive than ideal"]
        return 0, "GK review", ["Needs proactive GK traits"]

    if slot_key in {"LCB", "RCB"}:
        if "initiative" in role:
            bonus += 10; traits.append("Progresses possession")
        if "mobile" in role:
            bonus += 6; traits.append("Defends open space")
        if "physical" in role:
            bonus += 5; traits.append("Second-ball / aerial value")
        if "safe" in role:
            bonus += 2; traits.append("Possession security")
        fit_type = "Reinforcement fit" if bonus >= 8 else "Review"

    elif slot_key in {"LB", "RB"}:
        if "powerhouse" in role:
            bonus += 12; traits.extend(["High pressing lane", "Carries width", "Box arrival"])
            fit_type = "Reinforcement fit"
        elif "creative" in role:
            bonus += 10; traits.extend(["Short-passing support", "Final-third quality", "Interior access"])
            fit_type = "Evolution fit"
        elif "defensive" in role:
            bonus -= 4; traits.append("Balance option, less evolution")
            fit_type = "Fallback balance"

    elif slot_key in {"DMF-L", "DMF-R"}:
        if "progressor" in role:
            bonus += 12; traits.append("Short-passing progression")
        if "shielder" in role:
            bonus += 7; traits.append("Protects losses")
        if "recoverer" in role:
            bonus += 7; traits.append("Counterpress / second balls")
        if "box-to-box" in role or "box to box" in role:
            bonus += 8; traits.extend(["Vertical coverage", "Pressing support", "Second balls"])
        if "passing progressor" in role:
            bonus += 5; traits.append("Can support buildup")
        if "non progressor" in role:
            bonus -= 5; traits.append("May limit central progression")
        fit_type = "Reinforcement fit" if bonus >= 10 else "Control / balance"

    elif slot_key == "AMF":
        if "passing progressor" in role:
            bonus += 14; traits.extend(["Central penetration", "Low-block solution", "Final pass"])
            fit_type = "Evolution fit"
        elif "dribbling progressor" in role:
            bonus += 12; traits.extend(["Carries through middle", "Breaks pressure", "Low-block solution"])
            fit_type = "Evolution fit"
        elif "deep runner" in role:
            bonus += 8; traits.extend(["Box arrival", "Better receiving zones"])
            fit_type = "Evolution fit"

    elif slot_key == "LW":
        # The left winger slot is intentionally biased towards an interior,
        # associative profile: a wide 10 / Extra Midfielder who can add xA,
        # through-pass threat and central penetration in a possession-heavy team.
        if "extra midfielder" in role or "passing progressor" in role:
            bonus += 18; traits.extend(["Interior combinations", "xA / through-pass threat", "Central penetration"])
            fit_type = "Evolution fit"
        elif "dribbling progressor" in role:
            bonus += 14; traits.extend(["Carries inside", "Breaks low block", "Combines centrally"])
            fit_type = "Evolution fit"
        elif "deep runner" in role:
            bonus += 11; traits.extend(["Runs in behind", "Box reception", "Depth threat"])
            fit_type = "Evolution fit"
        elif "wide dribbler" in role:
            bonus += 6; traits.extend(["1v1 threat", "Needs final pass check", "Less interior than ideal"])
            fit_type = "Usable but less ideal"

    elif slot_key == "RW":
        if "wide dribbler" in role:
            bonus += 12; traits.extend(["1v1 threat", "Low-block solution", "Counterpressing lane"])
            fit_type = "Evolution fit"
        elif "extra midfielder" in role:
            bonus += 12; traits.extend(["Interior variety", "Short combinations", "Central access"])
            fit_type = "Evolution fit"
        elif "deep runner" in role:
            bonus += 14; traits.extend(["Runs in behind", "Box reception", "Transition threat"])
            fit_type = "Evolution fit"

    elif slot_key == "CF":
        if "deep runner" in role:
            bonus += 16; traits.extend(["Runs in behind", "Good-value box reception", "Presses first line"])
            fit_type = "Evolution fit"
        elif "complete forward" in role:
            bonus += 14; traits.extend(["Finishing + linking", "Pressing", "Carries threat"])
            fit_type = "Evolution fit"
        elif "target man" in role:
            bonus += 7; traits.extend(["Box reference", "Aerial option", "Second balls"])
            fit_type = "Reinforcement fit"
        elif "false nine" in role:
            bonus += 6; traits.extend(["Connects central play", "Interior combinations"])
            fit_type = "Evolution fit"

    elif slot_key == "BENCH":
        bonus = 0
        traits.append("Tactical alternative")
        fit_type = "Depth option"

    return bonus, fit_type, traits[:3]



def role_score_by_keyword(role_scores: list[dict], keyword: str) -> Optional[dict]:
    keyword = str(keyword).strip().lower()
    for result in role_scores or []:
        role_name = display_role_name(result.get("role_name", "")).lower()
        role_id = str(result.get("role_id", "")).lower()
        if keyword in role_name or keyword in role_id:
            return result
    return None


def goalkeeper_slot_fit(row: pd.Series, role_scores: list[dict], position_eval: dict) -> dict:
    """Specific GK logic for a possession/high-line national-team model.

    GK is the only slot where outfield/GK swaps remain impossible. But among real
    goalkeepers we should not over-penalise a reactive profile: proactive GK is
    prioritised, while reactive GK remains a usable fallback if the player is an
    actual goalkeeper.
    """
    if not position_eval.get("valid", True):
        return {
            "score": 5.0,
            "role": "—",
            "label": "Invalid slot",
            "type": "Impossible position",
            "traits": [position_eval.get("note", "Invalid GK slot")],
            "valid_position": False,
            "position_note": position_eval.get("note", "Invalid GK slot"),
            "position_penalty": position_eval.get("penalty", 100),
        }

    proactive = role_score_by_keyword(role_scores, "proactive")
    reactive = role_score_by_keyword(role_scores, "reactive")
    best = role_scores[0] if role_scores else None

    proactive_score = float(proactive.get("role_fit_score", np.nan)) if proactive else np.nan
    reactive_score = float(reactive.get("role_fit_score", np.nan)) if reactive else np.nan
    best_score = float(best.get("role_fit_score", np.nan)) if best else np.nan

    if proactive is not None:
        role_name = display_role_name(proactive.get("role_name", "Proactive goalkeeper"))
        # Proactive traits are tactically more valuable here than the raw role score alone.
        score_parts = []
        weights = []
        if pd.notna(proactive_score):
            score_parts.append(proactive_score); weights.append(0.72)
        if pd.notna(reactive_score):
            score_parts.append(reactive_score); weights.append(0.28)
        base = np.average(score_parts, weights=weights) if score_parts else best_score
        score = float(np.clip(base + 28, 0, 100)) if pd.notna(base) else 72.0
        # A real proactive GK is the target profile for this model. Do not let
        # limited samples make him look like a weak fit when the role profile is correct.
        score = max(score, 72.0)
        return {
            "score": score,
            "role": role_name,
            "label": fit_label_for_slot(score, True),
            "type": "Critical fit",
            "traits": ["High-line coverage", "Short build-up", "Sweeper behaviour"],
            "valid_position": True,
            "position_note": "Natural position",
            "position_penalty": 0,
        }

    if reactive is not None:
        role_name = display_role_name(reactive.get("role_name", "Reactive goalkeeper"))
        base = reactive_score if pd.notna(reactive_score) else best_score
        score = float(np.clip(base + 14, 0, 100)) if pd.notna(base) else 58.0
        # Reactive GK is a fallback, but should not be punished drastically.
        # It only lacks the proactive bonus.
        score = max(score, 58.0)
        return {
            "score": score,
            "role": role_name,
            "label": fit_label_for_slot(score, True),
            "type": "Fallback GK",
            "traits": ["Shot-stopping fallback", "Needs build-up check", "Less proactive than ideal"],
            "valid_position": True,
            "position_note": "Natural position",
            "position_penalty": 0,
        }

    role_name = display_role_name(best.get("role_name", "Goalkeeper")) if best else "Goalkeeper"
    score = float(np.clip(best_score, 0, 100)) if pd.notna(best_score) else 50.0
    score = max(score, 50.0)
    return {
        "score": score,
        "role": role_name,
        "label": fit_label_for_slot(score, True),
        "type": "GK review",
        "traits": ["Check proactive actions", "Check passing security", "Check aerial control"],
        "valid_position": True,
        "position_note": "Natural position",
        "position_penalty": 0,
    }


def natural_slot_for_bench(player: dict, row: pd.Series) -> str:
    """Choose the slot used to evaluate a bench player.

    Bench should not penalise a player for not being in the XI. Instead, evaluate
    him as depth for the role/slot he would most naturally cover.
    """
    inferred = infer_shadow_slots(player)
    for slot in inferred:
        if slot and slot != "BENCH":
            return slot

    codes = player_position_codes_for_slot_check(row)
    if "GK" in codes:
        return "GK"
    if codes.intersection({"CF", "ST", "STRIKER"}):
        return "CF"
    if codes.intersection({"LW", "LWF", "LMF", "AML", "LWM", "LF"}):
        return "LW"
    if codes.intersection({"RW", "RWF", "RMF", "AMR", "RWM", "RF"}):
        return "RW"
    if codes.intersection({"AMF", "LAMF", "RAMF", "AM"}):
        return "AMF"
    if codes.intersection({"DMF", "LDMF", "RDMF", "DM"}):
        return "DMF-L"
    if codes.intersection({"CMF", "LCMF", "RCMF", "CM"}):
        return "DMF-L"
    if codes.intersection({"CB", "LCB", "RCB"}):
        return "LCB"
    if codes.intersection({"LB", "LWB"}):
        return "LB"
    if codes.intersection({"RB", "RWB"}):
        return "RB"
    return "BENCH"


def bench_player_fit(player: dict, row: pd.Series, role_scores: list[dict]) -> dict:
    """Evaluate a bench player as depth for his natural role.

    Bench should not apply a tactical slot penalty. The card must explain what
    the rating refers to, e.g. "Bench AMF fit" or "Bench GK fit".
    """
    natural_slot = natural_slot_for_bench(player, row)
    natural_label = SHADOW_SLOT_LABELS.get(natural_slot, natural_slot)

    if natural_slot == "GK":
        fit = goalkeeper_slot_fit(row, role_scores, slot_position_penalty(row, "GK"))
        fit = dict(fit)
        score = fit.get("score", np.nan)
        fit["label"] = f"{natural_label} fit: {fit_label(score)}"
        fit["type"] = f"Bench depth: {natural_label}"
        fit["position_note"] = f"Rated as {natural_label}, no bench penalty"
        fit["position_penalty"] = 0
        traits = fit.get("traits", []) or []
        fit["traits"] = ([f"If used as {natural_label}"] + traits)[:3]
        return fit

    best = role_scores[0] if role_scores else None
    if best is None:
        return {
            "score": np.nan,
            "role": "—",
            "label": f"{natural_label} fit: Needs review",
            "type": f"Bench depth: {natural_label}",
            "traits": [f"If used as {natural_label}", "Depth option"],
            "valid_position": True,
            "position_note": f"Rated as {natural_label}, no bench penalty",
            "position_penalty": 0,
        }

    role_name = display_role_name(best.get("role_name", "—"))
    role_score = float(best.get("role_fit_score", np.nan))
    team_bonus, fit_type, tactical_traits = tactical_slot_bonus(role_name, natural_slot)
    role_alignment_bonus = 6 if role_matches_slot(role_name, natural_slot) else 0

    score = float(np.clip(role_score + team_bonus + role_alignment_bonus, 0, 100)) if pd.notna(role_score) else np.nan

    traits = [f"If used as {natural_label}"]
    traits.extend(tactical_traits or [f"Covers {natural_label}", "Depth option"])

    return {
        "score": score,
        "role": role_name,
        "label": f"{natural_label} fit: {fit_label(score)}",
        "type": f"Bench depth: {natural_label}",
        "traits": traits[:3],
        "valid_position": True,
        "position_note": f"Rated as {natural_label}, no bench penalty",
        "position_penalty": 0,
    }

def get_shadow_player_fit(player: dict, slot_key: str) -> dict:
    row = get_player_row_by_key(player.get("player_key"))
    if row is None:
        return {"score": np.nan, "role": "—", "label": "Needs review", "type": "Review", "traits": [], "valid_position": False, "position_note": "Player row not found", "position_penalty": np.nan}

    position_eval = slot_position_penalty(row, slot_key)

    role_scores = calculate_role_fit_scores(TOTAL_SEASON_DF, row)
    if not role_scores:
        base = 8 if not position_eval["valid"] else np.nan
        return {"score": base, "role": "—", "label": fit_label_for_slot(base, position_eval["valid"]), "type": "Invalid slot" if not position_eval["valid"] else "Review", "traits": [], "valid_position": position_eval["valid"], "position_note": position_eval["note"], "position_penalty": position_eval["penalty"]}

    if slot_key == "BENCH":
        return apply_selection_context_adjustment(bench_player_fit(player, row, role_scores), row)

    if slot_key == "GK":
        return apply_selection_context_adjustment(goalkeeper_slot_fit(row, role_scores, position_eval), row)

    best = role_scores[0]
    role_name = display_role_name(best.get("role_name", "—"))
    role_score = float(best.get("role_fit_score", np.nan))

    # Only GK/outfield swaps are hard invalid. Everything else is penalised, not blocked.
    if not position_eval["valid"]:
        score = 5.0
        return {
            "score": score,
            "role": role_name,
            "label": "Invalid slot",
            "type": "Impossible position",
            "traits": [position_eval["note"]],
            "valid_position": False,
            "position_note": position_eval["note"],
            "position_penalty": position_eval["penalty"],
        }

    role_match = role_matches_slot(role_name, slot_key)
    team_bonus, fit_type, tactical_traits = tactical_slot_bonus(role_name, slot_key)

    if role_match:
        role_alignment_bonus = 6
    else:
        # Do not over-punish natural midfield/forward adaptations.
        # The position penalty already captures the slot change; this is only a role-shape adjustment.
        level = str(position_eval.get("level", ""))
        if level == "natural":
            role_alignment_bonus = -6
        elif level == "adapted":
            role_alignment_bonus = -2
        elif level == "stretch":
            role_alignment_bonus = -5
        else:
            role_alignment_bonus = -8

    position_penalty = float(position_eval.get("penalty", 0) or 0)
    score = float(np.clip(role_score + team_bonus + role_alignment_bonus - position_penalty, 0, 100)) if pd.notna(role_score) else np.nan

    # If the player is from a compatible football family, avoid visually absurd scores
    # caused by the role template being calculated for his natural role rather than the assigned slot.
    if pd.notna(score) and position_eval.get("level") == "adapted":
        score = max(score, min(58.0, max(44.0, role_score - 8)))
    elif pd.notna(score) and position_eval.get("level") == "stretch":
        score = max(score, min(48.0, max(34.0, role_score - 18)))

    if position_penalty >= 28:
        fit_type = "Major role change"
    elif position_penalty >= 16:
        fit_type = "Adaptation risk"
    elif not role_match:
        fit_type = "Adapted role"

    slot_needs = SHADOW_SLOT_REQUIREMENTS.get(slot_key, {}).get("needs", [])
    traits = tactical_traits or slot_needs[:3]
    if position_penalty > 0 and len(traits) < 3:
        traits.append(f"Position penalty -{int(position_penalty)}")

    fit_result = {
        "score": score,
        "role": role_name,
        "label": fit_label_for_slot(score, True),
        "type": fit_type,
        "traits": traits[:3],
        "valid_position": True,
        "position_note": position_eval["note"],
        "position_penalty": position_penalty,
    }
    return apply_selection_context_adjustment(fit_result, row)


def empty_slot_need_card(slot_key: str, slot_label: str):
    req = SHADOW_SLOT_REQUIREMENTS.get(slot_key, SHADOW_SLOT_REQUIREMENTS["BENCH"])
    priority_color = {
        "Critical": DBU_RED_SOFT,
        "Very high": "#fb7185",
        "High": "#f97316",
        "Medium": "#eab308",
        "Flexible": DBU_MUTED,
    }.get(req.get("priority"), DBU_MUTED)

    return html.Div(
        [
            html.Div(slot_label, style={"fontSize": "10px", "fontWeight": "900", "letterSpacing": ".12em", "color": DBU_RED_SOFT}),
            html.Div(req["title"], style={"fontSize": "12px", "fontWeight": "950", "marginTop": "4px", "lineHeight": "1.12"}),
            html.Div(f"Priority: {req['priority']}", style={"fontSize": "10px", "color": priority_color, "fontWeight": "850", "marginTop": "5px"}),
            html.Div(
                [html.Div(f"• {need}", style={"fontSize": "10px", "color": DBU_MUTED, "lineHeight": "1.25"}) for need in req.get("needs", [])[:3]],
                style={"marginTop": "6px"},
            ),
        ],
        style={
            "background": "rgba(255,255,255,.025)",
            "border": f"1px dashed {DBU_BORDER}",
            "borderRadius": "14px",
            "padding": "9px 10px",
            "minWidth": "148px",
            "maxWidth": "170px",
            "textAlign": "center",
        },
    )


def tactical_context_panel():
    return html.Div(
        [
            html.Div("Team tactical profile", style={**S["label"], "margin": "0 0 8px"}),
            html.H3("Dominant 4-2-3-1", style={"margin": "0 0 6px", "fontSize": "20px"}),
            html.Div(
                "Short-passing, possession-based 4-2-3-1. The team progresses through sequences, presses high, can counter quickly, but needs better runs in behind, second-ball security and defensive set-piece control.",
                style={"color": DBU_MUTED, "fontSize": "12px", "lineHeight": "1.45"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(item["label"], style={"fontSize": "12px", "fontWeight": "900"}),
                            html.Div(item["why"], style={"fontSize": "10px", "color": DBU_MUTED, "lineHeight": "1.35", "marginTop": "3px"}),
                        ],
                        style={
                            "background": "rgba(255,255,255,.035)",
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "12px",
                            "padding": "9px",
                        },
                    )
                    for item in TEAM_NEEDS
                ],
                style={"display": "grid", "gap": "8px", "marginTop": "12px"},
            ),
            html.Div("Color code", style={**S["label"], "margin": "14px 0 8px"}),
            html.Div(
                [
                    html.Div([html.Span(style={"display": "inline-block", "width": "9px", "height": "9px", "borderRadius": "999px", "background": "#22c55e", "marginRight": "6px"}), "Strong fit"], style={"fontSize": "11px", "color": DBU_MUTED}),
                    html.Div([html.Span(style={"display": "inline-block", "width": "9px", "height": "9px", "borderRadius": "999px", "background": "#eab308", "marginRight": "6px"}), "Usable fit"], style={"fontSize": "11px", "color": DBU_MUTED}),
                    html.Div([html.Span(style={"display": "inline-block", "width": "9px", "height": "9px", "borderRadius": "999px", "background": "#ef4444", "marginRight": "6px"}), "Weak / adaptation risk"], style={"fontSize": "11px", "color": DBU_MUTED}),
                ],
                style={"display": "grid", "gap": "5px"},
            ),
        ],
        style={
            "background": "rgba(0,0,0,.16)",
            "border": "1px solid rgba(255,255,255,.06)",
            "borderRadius": "20px",
            "padding": "14px",
            "minWidth": "280px",
        },
    )

def shadow_slot_card(slot_key: str, slot_label: str, player: Optional[dict]):
    if player is None:
        return empty_slot_need_card(slot_key, slot_label)

    player_key = str(player.get("player_key", ""))
    assigned_slot = str(player.get("assigned_slot", slot_key if slot_key in SHADOW_FORMATION_SLOTS else "BENCH"))
    if assigned_slot not in SHADOW_FORMATION_SLOTS:
        assigned_slot = "BENCH"

    fit = get_shadow_player_fit(player, slot_key)
    fit_score = fit.get("score", np.nan)
    color = fit_color(fit_score)

    return html.Div(
        [
            html.Div(slot_label, style={"fontSize": "10px", "fontWeight": "900", "letterSpacing": ".12em", "color": DBU_RED_SOFT}),
            html.Div(player.get("player_name", "Unknown"), style={"fontSize": "13px", "fontWeight": "950", "marginTop": "3px", "lineHeight": "1.1"}),
            html.Div(f"{player.get('position', '-')} · {player.get('team', '-')}", style={"fontSize": "10px", "color": DBU_MUTED, "marginTop": "4px"}),
            html.Div(display_role_name(fit.get("role", "—")), style={"fontSize": "10px", "fontWeight": "850", "color": DBU_TEXT, "marginTop": "6px"}),
            html.Div(
                f"{fit.get('label', 'Needs review')}" + (f" · {fit_score:.0f}" if pd.notna(fit_score) else ""),
                style={
                    "fontSize": "9px",
                    "fontWeight": "900",
                    "color": color,
                    "border": f"1px solid {color}",
                    "borderRadius": "999px",
                    "padding": "2px 6px",
                    "display": "inline-block",
                    "marginTop": "6px",
                },
            ),
            html.Div(
                fit.get("type", ""),
                style={"fontSize": "9px", "color": DBU_MUTED, "fontWeight": "800", "marginTop": "5px", "lineHeight": "1.15"},
            ),
            html.Div(
                fit.get("position_note", ""),
                style={"fontSize": "9px", "color": "#ef4444" if not fit.get("valid_position", True) else DBU_MUTED, "fontWeight": "750", "marginTop": "3px", "lineHeight": "1.15"},
            ) if fit.get("position_note") and fit.get("position_note") not in {"Natural position", "Bench option"} else None,
            html.Div(
                [html.Div(f"• {trait}", style={"fontSize": "9px", "color": DBU_MUTED, "lineHeight": "1.18"}) for trait in fit.get("traits", [])[:3]],
                style={"marginTop": "5px"},
            ) if fit.get("traits") else None,
            html.Button(
                "Remove",
                id={"type": "remove-shadow-player", "player_id": player_key},
                n_clicks=0,
                style={
                    "background": "rgba(255,255,255,.045)",
                    "color": DBU_MUTED,
                    "border": f"1px solid {DBU_BORDER}",
                    "borderRadius": "999px",
                    "padding": "3px 8px",
                    "fontSize": "9px",
                    "fontWeight": "800",
                    "cursor": "pointer",
                    "marginTop": "7px",
                },
            ),
        ],
        style={
            "background": "linear-gradient(180deg, rgba(255,255,255,.05), rgba(212,0,0,.065))",
            "border": f"1px solid {color}",
            "borderRadius": "14px",
            "padding": "9px 10px",
            "minWidth": "148px",
            "textAlign": "center",
            "boxShadow": "0 6px 18px rgba(0,0,0,.20)",
        },
    )

def shadow_row(children, top_margin="8px"):
    return html.Div(children, style={"display": "flex", "justifyContent": "center", "gap": "10px", "marginTop": top_margin, "flexWrap": "wrap"})


def bench_player_card(player: dict):
    return shadow_slot_card("BENCH", "Bench", player)


def shadow_squad_pitch(shadow_squad: list[dict]):
    slots, bench = assign_shadow_squad_slots(shadow_squad)
    return html.Div(
        [
            shadow_row([shadow_slot_card("CF", "CF", slots["CF"])]),
            shadow_row([shadow_slot_card("LW", "LW", slots["LW"]), shadow_slot_card("AMF", "AMF", slots["AMF"]), shadow_slot_card("RW", "RW", slots["RW"])]),
            shadow_row([shadow_slot_card("DMF-L", "DMF L", slots["DMF-L"]), shadow_slot_card("DMF-R", "DMF R", slots["DMF-R"])]),
            shadow_row([shadow_slot_card("LB", "LB", slots["LB"]), shadow_slot_card("LCB", "LCB", slots["LCB"]), shadow_slot_card("RCB", "RCB", slots["RCB"]), shadow_slot_card("RB", "RB", slots["RB"])]),
            shadow_row([shadow_slot_card("GK", "GK", slots["GK"])]),
            html.Div(
                [
                    html.Div("Bench / depth", style={**S["label"], "margin": "12px 0 8px"}),
                    html.Div(
                        [bench_player_card(p) for p in bench]
                        or [html.Div("No bench players yet.", style={"color": DBU_MUTED, "fontSize": "12px"})],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                            "gap": "10px",
                        },
                    ),
                ],
                style={"marginTop": "10px"},
            ),
        ],
        style={
            "background": f"radial-gradient(circle at center, rgba(212,0,0,.18), transparent 55%), linear-gradient(180deg, {DBU_PITCH_TOP} 0%, {DBU_PITCH_MID} 55%, {DBU_PITCH_BOTTOM} 100%)",
            "border": f"1px solid rgba(212,0,0,.45)",
            "borderRadius": "22px",
            "padding": "16px 12px",
            "boxShadow": "inset 0 0 0 1px rgba(255,255,255,.04), 0 0 36px rgba(212,0,0,.10)",
        },
    )

def shadow_squad_panel(shadow_squad: list[dict]):
    count = len(shadow_squad or [])
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Shadow squad", style={**S["label"], "margin": "0 0 6px"}),
                            html.H2("4-2-3-1 squad builder", style={"margin": 0, "fontSize": "22px"}),
                            html.Div(f"{count} selected player{'s' if count != 1 else ''} · choose slot before adding from profile", style=S["subtitle"]),
                        ]
                    ),
                    html.Button(
                        "Clear squad",
                        id="clear_shadow_squad_btn",
                        n_clicks=0,
                        style={
                            "background": "rgba(255,255,255,.045)",
                            "color": DBU_TEXT,
                            "border": f"1px solid {DBU_BORDER}",
                            "borderRadius": "999px",
                            "padding": "8px 12px",
                            "fontWeight": "850",
                            "fontSize": "12px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "gap": "14px", "alignItems": "center", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(shadow_squad_pitch(shadow_squad or []), style={"minWidth": 0}),
                    tactical_context_panel(),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(620px, 1fr) 320px",
                    "gap": "14px",
                    "alignItems": "start",
                },
            ),
        ]
    )

# ============================================================
# APP
# ============================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-menu-outer, .Select-option, .VirtualizedSelectOption {
                background: #f5f5f0 !important;
                color: #111827 !important;
            }
            .Select-option:hover, .Select-option.is-focused, .VirtualizedSelectFocusedOption {
                background: #e5e7eb !important;
                color: #111827 !important;
            }
            .Select-value-label { color: #111827 !important; }
            .Select-control {
                background: #f5f5f0 !important;
                border: 1px solid #d1d5db !important;
            }
            .Select-placeholder { color: #6b7280 !important; }
            .shadow-add-dropdown .Select-control {
                background: #d40000 !important;
                border: 1px solid #ff4d4d !important;
                border-radius: 999px !important;
                box-shadow: 0 0 16px rgba(212,0,0,.25) !important;
                cursor: pointer !important;
            }
            .shadow-add-dropdown .Select-placeholder,
            .shadow-add-dropdown .Select-value-label {
                color: #f8fafc !important;
                font-weight: 900 !important;
                font-size: 12px !important;
            }
            .shadow-add-dropdown .Select-arrow {
                border-color: #f8fafc transparent transparent !important;
            }
            .shadow-add-dropdown .Select-menu-outer,
            .shadow-add-dropdown .Select-menu,
            .shadow-add-dropdown .Select-option,
            .shadow-add-dropdown .VirtualizedSelectOption {
                background: #8b0000 !important;
                color: #f8fafc !important;
                font-weight: 800 !important;
                border-color: #ff4d4d !important;
                z-index: 9999 !important;
            }
            .shadow-add-dropdown .Select-option:hover,
            .shadow-add-dropdown .Select-option.is-focused,
            .shadow-add-dropdown .VirtualizedSelectFocusedOption {
                background: #d40000 !important;
                color: #ffffff !important;
            }
            .shadow-add-dropdown * {
                color: #f8fafc !important;
            }
            input {
                border-radius: 8px;
                border: 1px solid #d1d5db;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""

app.layout = html.Div(
    [
        dcc.Store(id="pitch_positions_store", storage_type="memory", data=[]),
        dcc.Store(id="selected_player_store", storage_type="memory", data=None),
        dcc.Store(id="profile_mode_store", storage_type="memory", data="league"),
        dcc.Store(id="profile_metric_mode_store", storage_type="memory", data="per90"),
        dcc.Store(id="shadow_squad_store", storage_type="memory", data=[]),
        html.Div(
            [
                html.Div(
                    [
                        html.H1("TalentBlik ", style=S["title"]),
                        html.P("Modern Danish player scouting dashboard", style=S["subtitle"]),
                        html.Div("Player search", style=S["label"]),
                        dcc.Input(
                            id="search_player",
                            type="text",
                            value="",
                            placeholder="Search player...",
                            autoComplete="off",
                            persistence=False,
                            style={"width": "100%", "padding": "9px"},
                        ),
                        html.Div("League", style=S["label"]),
                        dcc.Dropdown(
                            id="league_filter",
                            options=[{"label": x, "value": x} for x in sorted(DF["league_unified"].dropna().unique())] if "league_unified" in DF.columns else [],
                            value=[],
                            multi=True,
                            placeholder="League",
                            persistence=False,
                        ),
                        html.Div("Club", style=S["label"]),
                        dcc.Dropdown(id="team_filter", options=[], value=[], multi=True, placeholder="Club", persistence=False),
                        html.Div(id="pitch_map_wrapper", children=build_pitch_map([])),
                        html.Div("Age range", style={**S["label"], "marginTop": "30px"}),
                        dcc.RangeSlider(
                            id="age_filter",
                            min=age_min,
                            max=age_max,
                            value=[age_min, age_max],
                            step=1,
                            persistence=False,
                            marks={i: {"label": str(i), "style": {"fontSize": "9px", "color": DBU_MUTED}} for i in range(age_min, age_max + 1, 5)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div("Minutes range", style={**S["label"], "marginTop": "38px"}),
                        dcc.RangeSlider(
                            id="minutes_range",
                            min=0,
                            max=5000,
                            step=500,
                            value=[0, 5000],
                            persistence=False,
                            marks={i: {"label": str(i), "style": {"fontSize": "8px", "color": DBU_MUTED}} for i in range(0, 5001, 500)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style=S["sidebar"],
                ),
                html.Div(
                    [
                        html.Div(id="dashboard_header_panel", style=S["panel"]),
                        html.Div(id="shadow_squad_panel", style=S["panel"]),
                        html.Div(id="summary_panel", style=S["panel"]),
                        html.Div(id="player_cards_panel", style=S["panel"]),
                        html.Div(id="profile_panel", style=S["panel"]),
                    ],
                    style=S["main"],
                ),
            ],
            style=S["shell"],
        ),
    ],
    style=S["page"],
)

# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("pitch_positions_store", "data"),
    Input({"type": "pitch-role", "role": ALL}, "n_clicks"),
    State({"type": "pitch-role", "role": ALL}, "id"),
    State("pitch_positions_store", "data"),
    prevent_initial_call=True,
)
def toggle_pitch_position(n_clicks_list, ids, selected):
    selected = selected or []
    triggered_id = callback_context.triggered_id

    # Dash can fire pattern-matching callbacks when the pitch buttons mount.
    # On mount, n_clicks are 0/None. Ignore those so CF is not selected by default.
    if triggered_id is None:
        raise PreventUpdate

    if not isinstance(triggered_id, dict):
        raise PreventUpdate

    if triggered_id.get("type") != "pitch-role":
        raise PreventUpdate

    role = triggered_id.get("role")
    if role is None:
        raise PreventUpdate

    clicked_n = 0
    for n_clicks, component_id in zip(n_clicks_list or [], ids or []):
        if component_id and component_id.get("role") == role:
            clicked_n = n_clicks or 0
            break

    if clicked_n <= 0:
        raise PreventUpdate

    if role in selected:
        selected.remove(role)
    else:
        selected.append(role)

    return selected


@app.callback(Output("pitch_map_wrapper", "children"), Input("pitch_positions_store", "data"))
def refresh_pitch_map(selected):
    return build_pitch_map(selected or [])


@app.callback(Output("team_filter", "options"), Output("team_filter", "value"), Input("league_filter", "value"))
def update_team_options(leagues):
    dff = DF.copy()
    if leagues and "league_unified" in dff.columns:
        dff = dff[dff["league_unified"].isin(leagues)]
    teams = sorted(dff["team_name_unified"].dropna().unique()) if "team_name_unified" in dff.columns else []
    return [{"label": t, "value": t} for t in teams], []


@app.callback(
    Output("selected_player_store", "data"),
    Input({"type": "player-card", "player_id": ALL}, "n_clicks"),
    Input("back_to_player_search_btn", "n_clicks", allow_optional=True),
    State({"type": "player-card", "player_id": ALL}, "id"),
    State("selected_player_store", "data"),
    prevent_initial_call=True,
)
def select_player_card(card_clicks, back_clicks, ids, current_selected_player):
    triggered_id = callback_context.triggered_id
    print("[selected_player_callback] triggered_id=", triggered_id, "current_selected_player=", current_selected_player)

    # IMPORTANT:
    # This callback is the ONLY place allowed to write selected_player_store.
    # It must not clear selection because components mounted/unmounted during profile toggle re-renders.

    if triggered_id == "back_to_player_search_btn":
        if back_clicks and back_clicks > 0:
            print("[selected_player_callback] clearing selected player via Back button")
            return None
        raise PreventUpdate

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "player-card":
        clicked_key = triggered_id.get("player_id")
        if clicked_key is None:
            raise PreventUpdate

        # Ignore initial/mount noise from pattern-matching callbacks.
        clicked_n = 0
        for n_clicks, component_id in zip(card_clicks or [], ids or []):
            if component_id and component_id.get("player_id") == clicked_key:
                clicked_n = n_clicks or 0
                break

        if clicked_n <= 0:
            raise PreventUpdate

        print("[selected_player_callback] selected player=", clicked_key)
        return str(clicked_key)

    # Any other trigger, including scope/view changes or layout re-renders,
    # must keep the selected player unchanged.
    raise PreventUpdate


@app.callback(
    Output("shadow_squad_store", "data"),
    Input("add_shadow_slot_dropdown", "value", allow_optional=True),
    Input({"type": "remove-shadow-player", "player_id": ALL}, "n_clicks"),
    Input("clear_shadow_squad_btn", "n_clicks", allow_optional=True),
    State("selected_player_store", "data"),
    State("shadow_squad_store", "data"),
    prevent_initial_call=True,
)
def update_shadow_squad(add_slot_value, remove_clicks, clear_clicks, selected_player_id, current_squad):
    current_squad = current_squad or []
    triggered_id = callback_context.triggered_id

    if triggered_id is None:
        raise PreventUpdate

    if triggered_id == "clear_shadow_squad_btn":
        if clear_clicks and clear_clicks > 0:
            return []
        raise PreventUpdate

    if triggered_id == "add_shadow_slot_dropdown":
        if not add_slot_value or selected_player_id is None:
            raise PreventUpdate

        row = get_player_row_by_key(selected_player_id)
        if row is None:
            raise PreventUpdate

        payload = shadow_player_payload(row)
        player_key = payload.get("player_key")
        selected_slot = add_slot_value if add_slot_value in SHADOW_FORMATION_SLOTS else "BENCH"

        updated_squad = []
        replaced_existing = False

        for player in current_squad:
            if str(player.get("player_key")) == str(player_key):
                player = dict(player)
                player["assigned_slot"] = selected_slot
                replaced_existing = True
            updated_squad.append(player)

        if not replaced_existing:
            payload["assigned_slot"] = selected_slot
            updated_squad.append(payload)

        return updated_squad

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "remove-shadow-player":
        player_key = str(triggered_id.get("player_id", ""))
        if not player_key:
            raise PreventUpdate
        return [p for p in current_squad if str(p.get("player_key")) != player_key]

    raise PreventUpdate


@app.callback(
    Output("shadow_squad_panel", "children"),
    Input("shadow_squad_store", "data"),
)
def render_shadow_squad_panel(shadow_squad):
    return shadow_squad_panel(shadow_squad or [])


@app.callback(
    Output("profile_mode_store", "data"),
    Input({"type": "profile-scope-btn", "value": ALL}, "n_clicks"),
    State("profile_mode_store", "data"),
    prevent_initial_call=True,
)
def update_profile_mode_store(scope_clicks, current_mode):
    if not any(scope_clicks or []):
        raise PreventUpdate

    triggered_id = callback_context.triggered_id

    if not isinstance(triggered_id, dict):
        raise PreventUpdate

    if triggered_id.get("type") != "profile-scope-btn":
        raise PreventUpdate

    value = triggered_id.get("value")

    if value not in {"league", "total"}:
        raise PreventUpdate

    return value or current_mode or "league"


@app.callback(
    Output("profile_metric_mode_store", "data"),
    Input({"type": "metric-view-btn", "value": ALL}, "n_clicks"),
    State("profile_metric_mode_store", "data"),
    prevent_initial_call=True,
)
def update_profile_metric_mode_store(metric_clicks, current_mode):
    if not any(metric_clicks or []):
        raise PreventUpdate

    triggered_id = callback_context.triggered_id

    if not isinstance(triggered_id, dict):
        raise PreventUpdate

    if triggered_id.get("type") != "metric-view-btn":
        raise PreventUpdate

    value = triggered_id.get("value")

    if value not in {"per90", "total"}:
        raise PreventUpdate

    return value or current_mode or "per90"


@app.callback(
    Output("dashboard_header_panel", "children"),
    Output("dashboard_header_panel", "style"),
    Output("summary_panel", "children"),
    Output("summary_panel", "style"),
    Output("player_cards_panel", "children"),
    Output("player_cards_panel", "style"),
    Output("profile_panel", "children"),
    Input("pitch_positions_store", "data"),
    Input("league_filter", "value"),
    Input("team_filter", "value"),
    Input("age_filter", "value"),
    Input("minutes_range", "value"),
    Input("search_player", "value"),
    Input("selected_player_store", "data"),
    Input("profile_mode_store", "data"),
    Input("profile_metric_mode_store", "data"),
)
def update_dashboard(pitch_positions, leagues, teams, age_range, minutes_range, search_text, selected_player_id, profile_mode_value, profile_metric_mode_value):
    expanded_pitch_positions = expand_pitch_positions(pitch_positions or [])
    dff = filter_df(DF, expanded_pitch_positions, leagues, teams, age_range, minutes_range, search_text)

    if "minutes_played" in dff.columns:
        dff = dff.sort_values("minutes_played", ascending=False, na_position="last")

    count = len(dff)
    avg_age = dff["age"].mean() if count and "age" in dff.columns else np.nan

    selected_row = None
    if selected_player_id is not None:
        # Read selected player from the full dataframe, not the filtered result.
        # This prevents profile/metric toggles from kicking the user back to search mode.
        selected_df = DF[DF["player_key"].astype(str) == str(selected_player_id)]
        if not selected_df.empty:
            selected_row = selected_df.iloc[0]

    selected_mode = selected_row is not None

    header = html.Div(
        [
            html.Div(
                [
                    html.H1("Scouting Dashboard", style={**S["title"], "fontSize": "22px" if selected_mode else "30px"}),
                    html.P(
                        "Selected player profile" if selected_mode else "Filter players, open profiles, and build scouting views.",
                        style=S["subtitle"],
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div([html.Strong(str(count)), html.Span(" players", style={"color": DBU_MUTED})]),
                    html.Div([html.Strong("—" if pd.isna(avg_age) else f"{avg_age:.1f}"), html.Span(" avg age", style={"color": DBU_MUTED})]),
                    html.Div([html.Strong(str(len(expanded_pitch_positions))), html.Span(" positions", style={"color": DBU_MUTED})]),
                ],
                style={
                    "display": "flex",
                    "gap": "18px",
                    "fontSize": "13px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                },
            ),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "gap": "18px",
        },
    )

    header_style = dict(S["panel"])
    if selected_mode:
        header_style.update({"padding": "12px 16px", "borderRadius": "18px"})

    summary = html.Div(
        [
            html.Div("Pool summary", style={**S["label"], "marginTop": "0"}),
            html.Div(
                [
                    html.Div([html.H2(str(count), style={"margin": 0}), html.Div("Players", style=S["subtitle"])]),
                    html.Div([html.H2("—" if pd.isna(avg_age) else f"{avg_age:.1f}", style={"margin": 0}), html.Div("Avg age", style=S["subtitle"])]),
                    html.Div([html.H2(str(len(expanded_pitch_positions)), style={"margin": 0}), html.Div("Position filters", style=S["subtitle"])]),
                ],
                style={"display": "flex", "gap": "34px"},
            ),
        ]
    )

    hidden_style = {"display": "none"}
    visible_panel_style = dict(S["panel"])

    if count == 0:
        return (
            header,
            header_style,
            summary,
            hidden_style if selected_mode else visible_panel_style,
            html.Div("No players match current filters."),
            visible_panel_style,
            player_profile(None, "league", "per90"),
        )

    if selected_row is not None:
        cards_panel = html.Div()
        player_cards_style = hidden_style
        summary_style = hidden_style
    else:
        cards_panel = html.Div(
            [
                html.Div("Filtered player cards", style={**S["label"], "marginTop": "0"}),
                html.Div([player_card(row, selected_player_id) for _, row in dff.head(36).iterrows()], style=S["cards"]),
            ]
        )
        player_cards_style = visible_panel_style
        summary_style = visible_panel_style

    profile = player_profile(selected_row, profile_mode_value or "league", profile_metric_mode_value or "per90")
    return header, header_style, summary, summary_style, cards_panel, player_cards_style, profile


if __name__ == "__main__":
    print("Loaded rows:", len(DF))
    print("Outfield League metric categories:", list(LEAGUE_METRIC_CATALOG.keys()))
    print("GK League metric categories:", list(LEAGUE_GK_METRIC_CATALOG.keys()))
    print("Wyscout Total metric categories:", list(WYSCOUT_METRIC_CATALOG.keys()))
    print("Role template metrics loaded:", len(ROLE_TEMPLATES))
    print("Role map rows loaded:", len(ROLE_MAP))
    print("Role template groups:", sorted(ROLE_TEMPLATES["position_group"].dropna().unique()) if not ROLE_TEMPLATES.empty else [])
    app.run(debug=True)
