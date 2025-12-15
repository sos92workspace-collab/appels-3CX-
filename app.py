"""Application Streamlit pour l'analyse des appels 3CX du standard SOS92.

L'application permet d'importer un ou plusieurs CSV 3CX, de les fusionner,
nettoyer et enrichir pour faciliter l'analyse par agent, période, direction
et statut. Tout tient dans ce fichier unique pour un lancement simple via
``streamlit run app.py``.
"""

import csv
import hashlib
import io
import json
import re
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st


PARIS_TZ = ZoneInfo("Europe/Paris")
EXPECTED_CARTE92_HEADER = [
    "Créé le",
    "Nom de la ville",
    "Action",
    "Secteur 92",
    "Standardiste",
]


# -------------------------
# Fonctions utilitaires
# -------------------------

def read_csv_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Lit un fichier CSV 3CX téléversé et retourne un DataFrame pandas.

    L'import tente plusieurs séparateurs et encodages pour être robuste face
    aux exports 3CX paramétrés en français (séparateur `;`) ou en anglais
    (séparateur `,`). En cas d'échec, un message explicite est affiché à
    l'utilisateur.
    """

    # Options de lecture testées en cascade :
    # - utf-8-sig pour retirer un éventuel BOM
    # - détection automatique du séparateur ou forcé en "," / ";"
    attempts = [
        {"encoding": "utf-8-sig", "sep": None},
        {"encoding": "utf-8", "sep": None},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin-1", "sep": ";"},
    ]

    last_error: Optional[Exception] = None

    for params in attempts:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine="python", **params)
        except Exception as err:  # pragma: no cover - affichage utilisateur
            last_error = err
            continue

    st.error(
        "Impossible de lire le fichier CSV (séparateur `,` ou `;`). "
        "Vérifiez le format et réessayez."
    )
    if last_error:
        st.error(f"Détail technique : {last_error}")
    return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes en supprimant les espaces superflus."""

    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    return df


def normalize_city_name(city: str) -> str:
    """Normalise le nom d'une ville pour les comparaisons insensibles à la casse."""

    normalized = " ".join(str(city).strip().split())
    return normalized.lower()


def extract_columns_from_bytes(file_bytes: bytes) -> List[str]:
    """Retourne la liste des colonnes d'un CSV en UTF-8-SIG sans consommer le buffer."""

    try:
        preview = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", sep=",", nrows=0)
    except Exception:
        return []
    return [str(col).strip().lstrip("\ufeff") for col in preview.columns]


def is_carte92_csv(file_bytes: bytes) -> bool:
    """Détermine si un fichier correspond exactement au format trafic_changement_carte_92."""

    columns = extract_columns_from_bytes(file_bytes)
    return columns == EXPECTED_CARTE92_HEADER


def parse_carte92_datetime(value: str) -> Optional[datetime]:
    """Parse une date au format DD/MM/YYYY HH:mm en timezone Europe/Paris."""

    text = str(value).strip()
    try:
        naive = datetime.strptime(text, "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=PARIS_TZ)


def build_carte92_event_id(created_at: datetime, city: str, action: str, sector_92: str, operator: str) -> str:
    """Construit la clé de déduplication deterministe pour un événement carte 92."""

    payload = f"{created_at.isoformat()}|{city}|{action}|{sector_92}|{operator}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def convert_duration_to_seconds(series: pd.Series) -> pd.Series:
    """Convertit une série de durées (HH:MM:SS, MM:SS ou nombre) en secondes."""

    def _to_seconds(value):
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text:
            return 0.0

        match = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.\d+)?$", text)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            return float(hours * 3600 + minutes * 60 + seconds)

        try:
            return float(text.replace(",", "."))
        except ValueError:
            return 0.0

    return series.apply(_to_seconds)


def extract_agent_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les noms/extensions depuis From/To en respectant la direction."""

    df = df.copy()
    pattern = re.compile(r"(?P<name>[^()]+?)\s*\((?P<ext>\d{2,})\)")

    def _parse_contact(value: str) -> pd.Series:
        match = pattern.search(str(value))
        if match:
            return pd.Series({
                "Name": match.group("name").strip(),
                "Ext": match.group("ext"),
            })
        return pd.Series({"Name": pd.NA, "Ext": pd.NA})

    from_info = df["From"].apply(_parse_contact)
    to_info = df["To"].apply(_parse_contact)

    df["FromName"] = from_info["Name"]
    df["FromExtension"] = from_info["Ext"]
    df["ToName"] = to_info["Name"]
    df["ToExtension"] = to_info["Ext"]

    def _select_agent(row):
        direction = str(row.get("Direction", "")).strip().lower()
        if direction.startswith("inbound"):
            return pd.Series({"AgentName": row["ToName"], "AgentExt": row["ToExtension"]})
        if direction.startswith("outbound"):
            return pd.Series({"AgentName": row["FromName"], "AgentExt": row["FromExtension"]})

        # Par défaut, on tente d'abord To puis From pour les appels internes/autres
        if pd.notna(row["ToExtension"]):
            return pd.Series({"AgentName": row["ToName"], "AgentExt": row["ToExtension"]})
        return pd.Series({"AgentName": row["FromName"], "AgentExt": row["FromExtension"]})

    agent_info = df.apply(_select_agent, axis=1)
    df["AgentName"] = agent_info["AgentName"]
    df["AgentExt"] = agent_info["AgentExt"]
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes dérivées nécessaires à l'analyse."""

    df = df.copy()
    # Conversion du Call Time en datetime
    df["Call Time"] = pd.to_datetime(df["Call Time"], errors="coerce")

    df["Date"] = df["Call Time"].dt.date
    df["Year"] = df["Call Time"].dt.year
    df["Month"] = df["Call Time"].dt.month
    df["Week"] = df["Call Time"].dt.isocalendar().week.astype("Int64")
    df["DayOfWeek"] = df["Call Time"].dt.day_name()
    df["Hour"] = df["Call Time"].dt.hour

    # Conversion des durées
    df["RingingSeconds"] = convert_duration_to_seconds(df.get("Ringing", pd.Series([0] * len(df))))
    df["TalkingSeconds"] = convert_duration_to_seconds(df.get("Talking", pd.Series([0] * len(df))))
    df["CallDurationSeconds"] = df["RingingSeconds"] + df["TalkingSeconds"]

    # Catégorisation CallType
    def categorize(row):
        direction = str(row.get("Direction", "")).strip().title()
        to_val = str(row.get("To", ""))
        details = str(row.get("Call Activity Details", ""))
        if "voicemail" in to_val.lower() or "voicemail" in details.lower():
            return "Voicemail"
        if "script" in to_val.lower() or "call script" in details.lower():
            return "Script"
        if direction in {"Inbound", "Outbound", "Internal"}:
            return direction
        return "Autre"

    df["CallType"] = df.apply(categorize, axis=1)
    return df


def to_numeric_extension(series: pd.Series) -> pd.Series:
    """Convertit une série d'extensions en nombres en ignorant les erreurs."""

    return pd.to_numeric(series, errors="coerce")


def build_aggregated_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Construit une table dédupliquée par Call ID avec agent décroché et file.

    Règles appliquées (identiques dans tout le code) :
    - Agent décroché : première ligne Status="Answered" dont l'extension extraite
      est comprise entre 100 et 130 (Standard), triée par Call Time croissant.
    - File retenue : dernière file (extension To >= 800 et != 992, ou Direction
      "Inbound Queue") rencontrée avant le décroché de l'agent. À défaut, la
      valeur "Sans file" est utilisée.
    - Déduplication : 1 ligne par couple (Call ID, agent) pour les appels
      décroché, ce qui évite de compter plusieurs fois un même appel.
    """

    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["AgentExtNum"] = to_numeric_extension(work["AgentExt"])
    work["ToExtensionNum"] = to_numeric_extension(work["ToExtension"])

    def _aggregate_call(call_id: str, group: pd.DataFrame):
        ordered = group.sort_values("Call Time")

        answered_agents = ordered[
            (ordered["Status"].str.lower() == "answered")
            & (ordered["AgentExtNum"].between(100, 130, inclusive="both"))
        ]
        if answered_agents.empty:
            return None

        # On retient le premier agent qui décroche (transfert géré en amont).
        agent_row = answered_agents.iloc[0]

        # Recherche de la file pertinente avant le décroché
        call_time = agent_row["Call Time"]
        queue_candidates = ordered[
            (
                (ordered["ToExtensionNum"] >= 800)
                & (ordered["ToExtensionNum"] != 992)
            )
            | (
                ordered["Direction"].str.strip().str.lower()
                == "inbound queue"
            )
        ]

        if pd.notna(call_time):
            queue_candidates = queue_candidates[queue_candidates["Call Time"] <= call_time]

        queue_candidates = queue_candidates.sort_values("Call Time")
        queue_value = "Sans file"
        if not queue_candidates.empty:
            last_queue = queue_candidates.iloc[-1]
            queue_value = last_queue.get("ToExtension") or queue_value

        return {
            "Call ID": call_id,
            "AgentExt": agent_row.get("AgentExt"),
            "AgentName": agent_row.get("AgentName"),
            "Direction": agent_row.get("Direction"),
            "Status": agent_row.get("Status"),
            "CallType": agent_row.get("CallType"),
            "Call Time": agent_row.get("Call Time"),
            "Date": agent_row.get("Date"),
            "Year": agent_row.get("Year"),
            "Month": agent_row.get("Month"),
            "Week": agent_row.get("Week"),
            "DayOfWeek": agent_row.get("DayOfWeek"),
            "Hour": agent_row.get("Hour"),
            "RingingSeconds": agent_row.get("RingingSeconds"),
            "TalkingSeconds": agent_row.get("TalkingSeconds"),
            "CallDurationSeconds": agent_row.get("CallDurationSeconds"),
            "RetainedQueue": queue_value,
        }

    records = []
    for call_id, group in work.groupby("Call ID"):
        aggregated = _aggregate_call(call_id, group)
        if aggregated:
            records.append(aggregated)

    return pd.DataFrame.from_records(records)


def ensure_carte92_state():
    """Initialise les structures de stockage pour les imports carte 92."""

    if "carte92_events" not in st.session_state:
        st.session_state.carte92_events = pd.DataFrame(
            columns=[
                "event_id",
                "created_at",
                "city",
                "normalized_city",
                "action",
                "sector_92",
                "operator",
                "imported_at",
                "source_file",
            ]
        )

    if "carte92_last_result" not in st.session_state:
        st.session_state.carte92_last_result = None


def parse_trafic_changement_carte_92(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> SimpleNamespace:
    """Traite un CSV trafic_changement_carte_92 et retourne un rapport d'import."""

    file_bytes = uploaded_file.getvalue()
    if not is_carte92_csv(file_bytes):
        return SimpleNamespace(handled=False, reason="header_mismatch")

    raw_df = pd.read_csv(
        io.BytesIO(file_bytes),
        encoding="utf-8-sig",
        sep=",",
        dtype=str,
        quotechar="\"",
        engine="python",
    )
    raw_df = normalize_columns(raw_df)

    errors: List[str] = []
    duplicates = 0
    records = []
    existing_ids = set(st.session_state.carte92_events.get("event_id", []))
    seen_ids = set(existing_ids)
    imported_at = datetime.now(PARIS_TZ)

    for idx, row in raw_df.iterrows():
        line_number = idx + 2  # +1 for header, +1 for 1-indexing
        created_at = parse_carte92_datetime(row.get("Créé le", ""))
        city = str(row.get("Nom de la ville", "")).strip()
        action = str(row.get("Action", "")).strip()
        sector_92 = str(row.get("Secteur 92", "")).strip()
        operator = str(row.get("Standardiste", "")).strip()

        if not created_at:
            errors.append(f"Ligne {line_number}: date invalide ({row.get('Créé le')}).")
            continue

        if not city:
            errors.append(f"Ligne {line_number}: ville manquante.")
            continue

        if action not in {"Ouverture", "Fermeture"}:
            errors.append(f"Ligne {line_number}: action inconnue ({action}).")
            continue

        event_id = build_carte92_event_id(created_at, city, action, sector_92, operator)
        if event_id in seen_ids:
            duplicates += 1
            continue

        seen_ids.add(event_id)
        records.append(
            {
                "event_id": event_id,
                "created_at": created_at,
                "city": city,
                "normalized_city": normalize_city_name(city),
                "action": action,
                "sector_92": sector_92,
                "operator": operator,
                "imported_at": imported_at,
                "source_file": uploaded_file.name,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        st.session_state.carte92_events = pd.concat(
            [st.session_state.carte92_events, df], ignore_index=True
        )

    result = SimpleNamespace(
        handled=True,
        total_rows=len(raw_df),
        inserted=len(df),
        duplicates=duplicates,
        errors=errors,
        source=uploaded_file.name,
    )
    st.session_state.carte92_last_result = result
    return result


def validate_required_columns(df: pd.DataFrame, filename: str) -> List[str]:
    """Vérifie la présence des colonnes indispensables et retourne la liste manquante."""

    required_cols = ["Call Time", "Call ID", "From", "To", "Direction", "Status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans {filename}: {', '.join(missing)}")
    return missing


def load_and_prepare_data(files) -> pd.DataFrame:
    """Charge plusieurs CSV, les concatène, supprime les doublons et enrichit les données."""

    frames: List[pd.DataFrame] = []

    for file in files:
        df = read_csv_file(file)
        df = normalize_columns(df)

        if validate_required_columns(df, file.name):
            continue

        df = extract_agent_info(df)
        df = add_derived_columns(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Supprime uniquement les doublons strictement identiques pour éviter de
    # perdre des lignes distinctes partageant le même Call ID ou horodatage
    # (plusieurs étapes d'un même appel, transferts, etc.).
    return combined.drop_duplicates().reset_index(drop=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les filtres utilisateurs sur le DataFrame."""

    if df.empty:
        return df

    st.sidebar.header("Filtres")

    # Période
    if df["Date"].notna().any():
        min_date, max_date = df["Date"].min(), df["Date"].max()
        start_date, end_date = st.sidebar.date_input(
            "Plage de dates",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    else:
        mask = pd.Series([True] * len(df))

    # Agents / extensions
    agent_names = st.sidebar.multiselect("Agents", sorted(df["AgentName"].dropna().unique()))
    if agent_names:
        mask &= df["AgentName"].isin(agent_names)

    agent_exts = st.sidebar.multiselect("Extensions", sorted(df["AgentExt"].dropna().unique()))
    if agent_exts:
        mask &= df["AgentExt"].isin(agent_exts)

    directions = st.sidebar.multiselect(
        "Direction", options=sorted(df["Direction"].dropna().unique())
    )
    if directions:
        mask &= df["Direction"].isin(directions)

    call_types = st.sidebar.multiselect(
        "Type d'appel", options=sorted(df["CallType"].dropna().unique())
    )
    if call_types:
        mask &= df["CallType"].isin(call_types)

    statuses = st.sidebar.multiselect(
        "Statut", options=sorted(df["Status"].dropna().unique())
    )
    if statuses:
        mask &= df["Status"].isin(statuses)

    # Filtre texte sur Call Activity Details (optionnel)
    if "Call Activity Details" in df.columns:
        search_text = st.sidebar.text_input("Recherche dans Call Activity Details")
        if search_text:
            mask &= df["Call Activity Details"].fillna("").str.contains(search_text, case=False, na=False)

    # Filtre sur durée
    min_dur = float(df.get("CallDurationSeconds", pd.Series([0])).min() or 0)
    max_dur = float(df.get("CallDurationSeconds", pd.Series([0])).max() or 0)
    duration_range = st.sidebar.slider(
        "Durée d'appel (secondes)",
        min_value=0.0,
        max_value=max(max_dur, 0.0),
        value=(min_dur, max_dur),
    )
    if "CallDurationSeconds" in df.columns:
        mask &= df["CallDurationSeconds"].between(duration_range[0], duration_range[1])

    return df[mask].copy()


def compute_kpis(df: pd.DataFrame):
    """Calcule les indicateurs clés pour le jeu de données filtré."""

    total_calls = len(df)
    answered_calls = (df["Status"].str.lower() == "answered").sum()
    missed_calls = df["Status"].str.lower().isin(["no answer", "missed"]).sum()
    total_talking = df["TalkingSeconds"].sum()
    total_ringing = df["RingingSeconds"].sum()
    avg_talking = df["TalkingSeconds"].mean() if total_calls else 0
    avg_ringing = df["RingingSeconds"].mean() if total_calls else 0
    distinct_agents = df["AgentExt"].nunique(dropna=True)

    cols = st.columns(6)
    cols[0].metric("Appels totaux", f"{total_calls}")
    cols[1].metric("Appels répondus", f"{answered_calls}")
    cols[2].metric("Appels manqués", f"{missed_calls}")
    cols[3].metric("Durée totale (talking)", f"{int(total_talking)} s")
    cols[4].metric("Durée totale (ringing)", f"{int(total_ringing)} s")
    cols[5].metric("Durée moyenne (ringing)", f"{avg_ringing:.1f} s")
    st.caption(
        f"Durée moyenne de conversation: {avg_talking:.1f} s · Agents distincts: {distinct_agents}"
    )


def stats_by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau de statistiques agrégées par agent."""

    if df.empty:
        return df

    grouped = df.groupby(["AgentExt", "AgentName"], dropna=False)
    result = grouped.agg(
        TotalCalls=("Call ID", "count"),
        InboundCalls=("Direction", lambda x: (x == "Inbound").sum()),
        OutboundCalls=("Direction", lambda x: (x == "Outbound").sum()),
        InternalCalls=("Direction", lambda x: (x == "Internal").sum()),
        AnsweredCalls=("Status", lambda x: (x.str.lower() == "answered").sum()),
        MissedCalls=("Status", lambda x: x.str.lower().isin(["no answer", "missed"]).sum()),
        TotalTalkingSeconds=("TalkingSeconds", "sum"),
        AvgTalkingSeconds=("TalkingSeconds", "mean"),
        TotalRingingSeconds=("RingingSeconds", "sum"),
        AvgRingingSeconds=("RingingSeconds", "mean"),
    ).reset_index()
    return result


def stats_by_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les volumes par file retenue (exclusion 992)."""

    if df.empty or "RetainedQueue" not in df.columns:
        return pd.DataFrame()

    filtered = df[df["RetainedQueue"].notna() & (df["RetainedQueue"] != "992")]
    return (
        filtered.groupby("RetainedQueue")
        .agg(AnsweredCalls=("Call ID", "count"))
        .reset_index()
        .sort_values("AnsweredCalls", ascending=False)
    )


def render_time_charts(df: pd.DataFrame):
    """Affiche des graphiques de répartition temporelle."""

    if df.empty:
        st.info("Aucune donnée filtrée pour afficher des graphiques.")
        return

    st.subheader("Répartition temporelle")
    calls_by_date = df.groupby("Date").size().reset_index(name="Count")
    chart_date = alt.Chart(calls_by_date).mark_line(point=True).encode(
        x="Date:T", y="Count:Q"
    )

    calls_by_hour = df.groupby("Hour").size().reset_index(name="Count")
    chart_hour = alt.Chart(calls_by_hour).mark_bar().encode(
        x="Hour:O", y="Count:Q"
    )

    dow_type = pd.CategoricalDtype(
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    calls_by_dow = df.copy()
    calls_by_dow["DayOfWeek"] = calls_by_dow["DayOfWeek"].astype(dow_type)
    calls_by_dow = calls_by_dow.groupby("DayOfWeek").size().reset_index(name="Count")
    chart_dow = alt.Chart(calls_by_dow).mark_bar().encode(
        x=alt.X("DayOfWeek", sort=list(dow_type.categories)),
        y="Count:Q",
    )

    st.altair_chart(chart_date, use_container_width=True)
    st.altair_chart(chart_hour, use_container_width=True)
    st.altair_chart(chart_dow, use_container_width=True)


def export_data(df: pd.DataFrame):
    """Propose un bouton pour télécharger les données filtrées au format CSV/Excel."""

    if df.empty:
        return

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Exporter les données filtrées (CSV)",
        data=csv_buffer.getvalue(),
        file_name="appels_filtres.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button(
        label="Exporter les données filtrées (Excel)",
        data=excel_buffer.getvalue(),
        file_name="appels_filtres.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_file_summary(files: Iterable, df: pd.DataFrame) -> None:
    """Affiche un résumé des fichiers chargés et de la période couverte."""

    file_names = ", ".join(f.name for f in files)
    st.success(f"{len(files)} fichier(s) chargé(s) – {len(df)} lignes après suppression des doublons.")
    st.caption(f"Fichiers importés : {file_names}")

    if "Call Time" in df.columns and df["Call Time"].notna().any():
        st.caption(f"Période couverte : {df['Call Time'].min().date()} → {df['Call Time'].max().date()}")


def ensure_import_state():
    """Prépare le stockage des imports dans la session Streamlit."""

    if "imports" not in st.session_state:
        st.session_state.imports = []


def register_import(file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    """Ajoute un fichier importé dans l'état de session avec ses métadonnées."""

    file_bytes = file.getvalue()
    buffer = io.BytesIO(file_bytes)
    buffer.name = file.name

    prepared_df = load_and_prepare_data([buffer])
    if prepared_df.empty:
        st.warning(f"{file.name} n'a pas pu être chargé et a été ignoré.")
        return

    call_times = prepared_df["Call Time"].dropna()
    period_min = call_times.min().date() if not call_times.empty else None
    period_max = call_times.max().date() if not call_times.empty else None

    st.session_state.imports.append(
        {
            "id": str(uuid.uuid4()),
            "name": file.name,
            "data": file_bytes,
            "uploaded_at": datetime.now(),
            "period_min": period_min,
            "period_max": period_max,
        }
    )
    st.success(f"{file.name} importé avec succès.")


def build_data_from_imports(import_entries: List[dict]) -> pd.DataFrame:
    """Recharge les fichiers importés pour produire le DataFrame complet."""

    if not import_entries:
        return pd.DataFrame()

    buffers: List[io.BytesIO] = []
    for entry in import_entries:
        buffer = io.BytesIO(entry["data"])
        buffer.name = entry["name"]
        buffers.append(buffer)

    return load_and_prepare_data(buffers)


@st.cache_data
def load_geojson_92(path: str) -> dict:
    """Charge le GeoJSON des communes du 92."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        st.error("GeoJSON des communes 92 introuvable.")
    except json.JSONDecodeError:
        st.error("GeoJSON invalide pour les communes du 92.")
    return {}


def compute_open_percentage_for_city(
    city_events: pd.DataFrame, start_dt: datetime, end_dt: datetime
) -> dict:
    """Calcule les métriques d'ouverture pour une ville donnée."""

    total_interval = (end_dt - start_dt).total_seconds()
    total_interval = max(total_interval, 0)

    ordered = city_events.sort_values("created_at") if not city_events.empty else pd.DataFrame()
    total_events = len(ordered)
    prior_events = ordered[ordered["created_at"] < start_dt] if not ordered.empty else pd.DataFrame()
    has_prior = not prior_events.empty
    last_prior = prior_events.iloc[-1] if has_prior else None
    initial_state = "open" if last_prior is not None and last_prior["action"] == "Ouverture" else "closed"

    window_events = ordered[
        (ordered["created_at"] >= start_dt) & (ordered["created_at"] <= end_dt)
    ]

    current_state_open = initial_state == "open"
    current_time = start_dt
    open_seconds = 0.0

    for _, event in window_events.iterrows():
        event_time = event["created_at"]
        if current_state_open:
            open_seconds += (event_time - current_time).total_seconds()

        current_state_open = event["action"] == "Ouverture"
        current_time = event_time

    if current_state_open and total_interval > 0:
        open_seconds += (end_dt - current_time).total_seconds()

    percent = (open_seconds / total_interval) if total_interval > 0 else 0.0

    return {
        "percent_open": percent,
        "duration_open": open_seconds,
        "total_interval": total_interval,
        "event_count": len(window_events),
        "initial_state": initial_state,
        "has_prior": has_prior,
        "total_events": total_events,
    }


def build_carte92_metrics(events_df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> dict:
    """Construit les métriques d'ouverture pour chaque ville importée."""

    metrics = {}
    if events_df.empty:
        return metrics

    for normalized_city, group in events_df.groupby("normalized_city"):
        metrics[normalized_city] = compute_open_percentage_for_city(group, start_dt, end_dt)
    return metrics


def color_from_percent(percent: Optional[float]) -> List[int]:
    """Retourne une couleur RGBA allant du rouge (0%) au vert (100%)."""

    if percent is None:
        return [180, 180, 180, 120]

    percent = max(0.0, min(1.0, percent))
    red = int(220 * (1 - percent))
    green = int(190 * percent)
    return [red, green, 80, 160]


def format_duration(seconds: float) -> str:
    """Formate une durée en secondes en HH:MM."""

    if seconds <= 0:
        return "0h00"
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}"


def prepare_geojson_features(
    geojson_data: dict, metrics: dict, total_interval: float
) -> Tuple[list, set]:
    """Injecte les métriques dans les features GeoJSON et retourne aussi la liste des villes non matchées."""

    features = geojson_data.get("features", []) if geojson_data else []
    unmatched_metrics = set(metrics.keys())
    enriched_features = []

    for feature in features:
        properties = feature.get("properties", {})
        city_name = properties.get("nom") or properties.get("name") or ""
        normalized_name = normalize_city_name(city_name)
        metric = metrics.get(
            normalized_name,
            {
                "percent_open": 0.0,
                "duration_open": 0.0,
                "total_interval": total_interval,
                "event_count": 0,
                "initial_state": "closed",
                "has_prior": False,
                "total_events": 0,
            },
        )

        if normalized_name in unmatched_metrics:
            unmatched_metrics.remove(normalized_name)

        percent = metric.get("percent_open")
        fill_color = color_from_percent(percent if metric.get("event_count") is not None else None)
        tooltip_lines = [f"Ouverture : {round((percent or 0) * 100)}%"]

        total_events = metric.get("total_events", metric.get("event_count", 0))
        if total_events == 0:
            tooltip_lines.append("Aucun événement")
        else:
            if total_interval == 0:
                tooltip_lines.append("Durée totale nulle")
            else:
                open_label = format_duration(metric.get("duration_open", 0.0))
                total_label = format_duration(metric.get("total_interval", total_interval))
                tooltip_lines.append(f"Ouvert {open_label} / {total_label}")

            tooltip_lines.append(f"{metric.get('event_count', 0)} événement(s)")
            if not metric.get("has_prior"):
                tooltip_lines.append("État initial supposé fermé")

        enriched_feature = feature.copy()
        enriched_properties = {**properties}
        enriched_properties.update(
            {
                "percent_open": percent,
                "percent_label": f"{round((percent or 0) * 100)}%",
                "fill_color": fill_color,
                "tooltip_detail": "<br/>".join(tooltip_lines),
            }
        )
        enriched_feature["properties"] = enriched_properties
        enriched_features.append(enriched_feature)

    return enriched_features, unmatched_metrics


def render_import_tab():
    """Affiche l'onglet Import pour téléverser et gérer les fichiers."""

    st.subheader("1) Importer les CSV")
    uploaded_files = st.file_uploader(
        "Choisissez un ou plusieurs fichiers CSV", type="csv", accept_multiple_files=True, key="import_uploader"
    )

    if uploaded_files:
        for file in uploaded_files:
            register_import(file)

    st.subheader("Historique des imports")
    if not st.session_state.imports:
        st.info("Aucun fichier importé pour le moment.")
        return

    header_cols = st.columns([2, 4, 3, 1])
    header_cols[0].markdown("**Date d'import**")
    header_cols[1].markdown("**Nom du fichier**")
    header_cols[2].markdown("**Période des données**")
    header_cols[3].markdown("**Actions**")

    for entry in list(st.session_state.imports):
        date_text = entry["uploaded_at"].strftime("%d/%m/%Y %H:%M")
        period_text = (
            f"{entry['period_min'].strftime('%d/%m/%Y')} → {entry['period_max'].strftime('%d/%m/%Y')}"
            if entry["period_min"] and entry["period_max"]
            else "Période inconnue"
        )

        cols = st.columns([2, 4, 3, 1])
        cols[0].write(date_text)
        cols[1].write(entry["name"])
        cols[2].write(period_text)
        if cols[3].button("✖", key=f"delete_{entry['id']}"):
            st.session_state.imports = [imp for imp in st.session_state.imports if imp["id"] != entry["id"]]
            st.success(f"Import {entry['name']} supprimé.")
            st.experimental_rerun()


def render_pilotage_tab():
    """Affiche l'onglet Pilotage avec tous les indicateurs."""

    if not st.session_state.imports:
        st.info("Aucun fichier importé pour le moment. Ajoutez des CSV dans l'onglet Import.")
        return

    data = build_data_from_imports(st.session_state.imports)
    if data.empty:
        st.warning("Impossible de charger les données. Vérifiez le format des fichiers.")
        return

    render_file_summary([SimpleNamespace(name=imp["name"]) for imp in st.session_state.imports], data)

    aggregated_calls = build_aggregated_calls(data)
    if aggregated_calls.empty:
        st.warning(
            "Aucun appel décroché par les agents du standard (extensions 100–130) "
            "n'a été identifié. Vérifiez le statut Answered dans vos exports."
        )
        return

    st.info(
        "Agrégation par Call ID : premier agent Standard qui décroche, "
        "dernière file rencontrée avant décroché (exclusion de la file 992)."
    )

    st.subheader("Indicateurs clés (appels décroché)")
    compute_kpis(aggregated_calls)

    st.subheader("Statistiques par agent (appels décroché)")
    st.dataframe(stats_by_agent(aggregated_calls))

    st.subheader("Répartition par files d'attente (global)")
    st.dataframe(stats_by_queue(aggregated_calls))

    render_time_charts(aggregated_calls)

    filtered = apply_filters(aggregated_calls)

    st.subheader("Indicateurs clés (données filtrées)")
    compute_kpis(filtered)

    st.subheader("Statistiques par agent (données filtrées)")
    st.dataframe(stats_by_agent(filtered))

    st.subheader("Répartition par files d'attente (données filtrées)")
    st.dataframe(stats_by_queue(filtered))

    render_time_charts(filtered)

    st.subheader("Données détaillées (appels agrégés après filtres)")
    st.dataframe(filtered.head(500))

    export_data(filtered)


def render_carte92_import_summary(result: Optional[SimpleNamespace]):
    """Affiche le résumé d'un import carte 92."""

    if not result or not getattr(result, "handled", False):
        return

    cols = st.columns(3)
    cols[0].metric("Lignes lues", result.total_rows)
    cols[1].metric("Enregistrées", result.inserted)
    cols[2].metric("Doublons", result.duplicates)

    if result.errors:
        st.error(f"{len(result.errors)} ligne(s) en erreur")
        for err in result.errors:
            st.caption(err)
    else:
        st.success("Aucune erreur détectée sur ce fichier.")


def render_import_carte92_tab():
    """Interface dédiée à l'import trafic_changement_carte_92."""

    ensure_carte92_state()
    st.subheader("Importer un CSV trafic_changement_carte_92")
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier CSV (UTF-8, séparateur virgule)",
        type="csv",
        key="carte92_uploader",
    )

    if st.button("Importer", type="primary"):
        if not uploaded_file:
            st.warning("Aucun fichier sélectionné.")
        else:
            result = parse_trafic_changement_carte_92(uploaded_file)
            if not result.handled:
                st.warning(
                    "En-tête non conforme : ce fichier a été ignoré afin de laisser les autres imports le traiter."
                )
            else:
                st.success(f"Import de {uploaded_file.name} terminé.")
                render_carte92_import_summary(result)

    if st.session_state.carte92_last_result:
        with st.expander("Dernier import carte 92"):
            render_carte92_import_summary(st.session_state.carte92_last_result)

    st.subheader("Événements enregistrés")
    total_events = len(st.session_state.carte92_events)
    st.caption(f"{total_events} événement(s) uniques enregistrés.")
    if total_events:
        st.dataframe(
            st.session_state.carte92_events.sort_values("created_at", ascending=False).head(200)
        )
    else:
        st.info("Aucun événement importé pour le moment.")


def render_ville92_tab():
    """Affiche la page Ville 92 avec carte, filtres et calculs d'ouverture."""

    ensure_carte92_state()
    st.subheader("Ville 92")
    st.caption(
        "Analyse des événements `trafic_changement_carte_92_events` : taux d'ouverture/fermeture par commune "
        "des Hauts-de-Seine sur une fenêtre filtrée."
    )

    events = st.session_state.carte92_events.copy()
    if events.empty:
        st.info("Importez d'abord des événements via l'onglet Import carte 92.")
        return

    events["created_at"] = pd.to_datetime(events["created_at"], utc=True).dt.tz_convert(PARIS_TZ)

    min_date = events["created_at"].min().date()
    max_date = events["created_at"].max().date()

    default_start = st.session_state.get("pilotage92_start", min_date)
    default_end = st.session_state.get("pilotage92_end", max_date)

    if "ville92_filters" not in st.session_state:
        st.session_state.ville92_filters = {
            "start_date": default_start,
            "end_date": default_end,
            "start_time": datetime.strptime("00:00", "%H:%M").time(),
            "end_time": datetime.strptime("23:59", "%H:%M").time(),
        }

    st.markdown("**Toutes les dates/heures sont interprétées en Europe/Paris.**")
    with st.form("ville_92_filters"):
        col1, col2 = st.columns(2)
        start_date = col1.date_input(
            "Date de début",
            value=st.session_state.ville92_filters["start_date"],
            min_value=min_date,
            max_value=max_date,
        )
        end_date = col2.date_input(
            "Date de fin",
            value=st.session_state.ville92_filters["end_date"],
            min_value=min_date,
            max_value=max_date,
        )

        col3, col4 = st.columns(2)
        start_time = col3.time_input(
            "Heure de début",
            value=st.session_state.ville92_filters["start_time"],
        )
        end_time = col4.time_input(
            "Heure de fin",
            value=st.session_state.ville92_filters["end_time"],
        )

        apply_filters_button = st.form_submit_button("Appliquer")
        reset_button = st.form_submit_button("Réinitialiser")

    if reset_button:
        st.session_state.ville92_filters = {
            "start_date": min_date,
            "end_date": max_date,
            "start_time": datetime.strptime("00:00", "%H:%M").time(),
            "end_time": datetime.strptime("23:59", "%H:%M").time(),
        }
        st.experimental_rerun()

    if apply_filters_button:
        proposed_filters = {
            "start_date": start_date,
            "end_date": end_date,
            "start_time": start_time,
            "end_time": end_time,
        }
        start_dt_test = datetime.combine(start_date, start_time, tzinfo=PARIS_TZ)
        end_dt_test = datetime.combine(end_date, end_time, tzinfo=PARIS_TZ)
        if end_dt_test < start_dt_test:
            st.error("La date/heure de fin doit être postérieure ou égale au début.")
        else:
            st.session_state.ville92_filters = proposed_filters

    start_date = st.session_state.ville92_filters["start_date"]
    end_date = st.session_state.ville92_filters["end_date"]
    start_time = st.session_state.ville92_filters["start_time"]
    end_time = st.session_state.ville92_filters["end_time"]

    st.session_state.pilotage92_start = start_date
    st.session_state.pilotage92_end = end_date

    start_dt = datetime.combine(start_date, start_time, tzinfo=PARIS_TZ)
    end_dt = datetime.combine(end_date, end_time, tzinfo=PARIS_TZ)

    if end_dt < start_dt:
        st.error("La date/heure de fin doit être postérieure ou égale au début.")
        return

    metrics = build_carte92_metrics(events, start_dt, end_dt)
    total_interval = max((end_dt - start_dt).total_seconds(), 0)

    geojson_data = load_geojson_92("assets/communes-hauts-de-seine.geojson")
    features, unmatched_metrics = prepare_geojson_features(geojson_data, metrics, total_interval)

    if not features:
        st.warning("Aucune donnée cartographique disponible.")
        return

    layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": features},
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color=[60, 60, 60, 180],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=48.85, longitude=2.25, zoom=10.5)
    tooltip = {
        "html": "<b>{nom}</b><br/>{percent_label}<br/>{tooltip_detail}",
        "style": {"backgroundColor": "white", "color": "#111"},
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    st.subheader("Synthèse par ville")
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        name = props.get("nom") or props.get("name") or "Inconnue"
        normalized_name = normalize_city_name(name)
        metric = metrics.get(
            normalized_name,
            {
                "percent_open": 0.0,
                "duration_open": 0.0,
                "event_count": 0,
                "has_prior": False,
                "total_events": 0,
            },
        )
        comment = ""
        if metric.get("total_events", 0) == 0:
            comment = "Aucun événement"
        elif not metric.get("has_prior"):
            comment = "Données initiales supposées fermées"
        rows.append(
            {
                "Ville": name,
                "% ouverture": round(metric.get("percent_open", 0) * 100),
                "Événements": metric.get("event_count", 0),
                "Durée ouverte": format_duration(metric.get("duration_open", 0.0)),
                "Remarque": comment or "",
            }
        )

    st.dataframe(pd.DataFrame(rows))

    if unmatched_metrics:
        st.warning(
            "Certaines villes importées n'ont pas été trouvées dans le GeoJSON : "
            + ", ".join(sorted(unmatched_metrics))
        )


# -------------------------
# Interface principale
# -------------------------

def main():
    st.set_page_config(page_title="Analyse des appels 3CX – SOS92", layout="centered")
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 900px;
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Analyse des appels 3CX – SOS92")
    st.write("Importez un ou plusieurs fichiers CSV 3CX pour analyser l'activité du standard.")

    ensure_import_state()
    ensure_carte92_state()
    tabs = st.tabs(["Pilotage", "Import", "Import carte 92", "Ville 92"])

    with tabs[1]:
        render_import_tab()

    with tabs[0]:
        render_pilotage_tab()

    with tabs[2]:
        render_import_carte92_tab()

    with tabs[3]:
        render_ville92_tab()


if __name__ == "__main__":
    main()
