"""Application Streamlit pour l'analyse des appels 3CX du standard SOS92.

L'application permet d'importer un ou plusieurs CSV 3CX, de les fusionner,
nettoyer et enrichir pour faciliter l'analyse par agent, période, direction
et statut. Tout tient dans ce fichier unique pour un lancement simple via
``streamlit run app.py``.
"""

import io
import re
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st


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

    for entry in list(st.session_state.imports):
        date_text = entry["uploaded_at"].strftime("%d/%m/%Y %H:%M")
        period_text = (
            f"{entry['period_min']} → {entry['period_max']}"
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


# -------------------------
# Interface principale
# -------------------------

def main():
    st.set_page_config(page_title="Analyse des appels 3CX – SOS92", layout="centered")
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Analyse des appels 3CX – SOS92")
    st.write("Importez un ou plusieurs fichiers CSV 3CX pour analyser l'activité du standard.")

    ensure_import_state()
    tabs = st.tabs(["Pilotage", "Import"])

    with tabs[1]:
        render_import_tab()

    with tabs[0]:
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


if __name__ == "__main__":
    main()
