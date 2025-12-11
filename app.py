"""Application Streamlit pour l'analyse des appels 3CX du standard SOS92.

L'application permet d'importer un ou plusieurs CSV 3CX, de les fusionner,
nettoyer et enrichir pour faciliter l'analyse par agent, période, direction
et statut. Tout tient dans ce fichier unique pour un lancement simple via
``streamlit run app.py``.
"""

import io
import re
from typing import Iterable, List

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

    def _try_read(*, sep, encoding):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=sep, engine="python", encoding=encoding)

    # Tente automatiquement de détecter le séparateur puis bascule sur les
    # séparateurs les plus fréquents. Deux encodages sont essayés pour éviter
    # les erreurs liées aux accents.
    attempts = [
        {"sep": None, "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": None, "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "latin-1"},
    ]

    for params in attempts:
        try:
            return _try_read(**params)
        except UnicodeDecodeError:
            # Passe à l'encodage suivant
            continue
        except pd.errors.ParserError:
            # Essaye une autre combinaison séparateur/encodage
            continue

    st.error(
        "Impossible de lire le fichier CSV. Vérifiez le séparateur (`,` ou `;`) "
        "et réessayez."
    )
    return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes en supprimant les espaces superflus."""

    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    return df


def convert_duration_to_seconds(series: pd.Series) -> pd.Series:
    """Convertit une série de durées au format HH:MM:SS en secondes."""

    def _to_seconds(value):
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        match = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.\d+)?$", text)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            return hours * 3600 + minutes * 60 + seconds

        try:
            return float(text)
        except ValueError:
            return 0

    return series.apply(_to_seconds)


def extract_agent_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le nom et l'extension d'agent depuis les colonnes From/To."""

    df = df.copy()
    pattern = re.compile(r"(?P<name>[^()]+?)\s*\((?P<ext>\d{2,})\)")

    def _extract(row):
        for field in ["From", "To"]:
            value = row.get(field, "")
            match = pattern.search(str(value))
            if match:
                name = match.group("name").strip()
                ext = match.group("ext")
                return pd.Series({"AgentName": name, "AgentExt": ext})
        return pd.Series({"AgentName": pd.NA, "AgentExt": pd.NA})

    agent_info = df.apply(_extract, axis=1)
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
    if "Call ID" in combined.columns:
        combined = combined.drop_duplicates(subset=["Call ID"], keep="first")
    return combined


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

    # Filtre texte sur Call Activity Details
    search_text = st.sidebar.text_input("Recherche dans Call Activity Details")
    if search_text:
        mask &= df["Call Activity Details"].fillna("").str.contains(search_text, case=False, na=False)

    # Filtre sur durée
    min_dur = float(df["CallDurationSeconds"].min() or 0)
    max_dur = float(df["CallDurationSeconds"].max() or 0)
    duration_range = st.sidebar.slider(
        "Durée d'appel (secondes)",
        min_value=0.0,
        max_value=max(max_dur, 0.0),
        value=(min_dur, max_dur),
    )
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


# -------------------------
# Interface principale
# -------------------------

def main():
    st.set_page_config(page_title="Analyse des appels 3CX – SOS92", layout="wide")
    st.title("Analyse des appels 3CX – SOS92")
    st.write("Importez un ou plusieurs fichiers CSV 3CX pour analyser l'activité du standard.")

    uploaded_files = st.file_uploader(
        "Choisissez un ou plusieurs fichiers CSV",
        type="csv",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Aucun fichier importé pour le moment.")
        return

    data = load_and_prepare_data(uploaded_files)
    if data.empty:
        st.warning("Impossible de charger les données. Vérifiez le format des fichiers.")
        return

    render_file_summary(uploaded_files, data)

    filtered = apply_filters(data)

    st.subheader("Indicateurs clés")
    compute_kpis(filtered)

    st.subheader("Statistiques par agent")
    st.dataframe(stats_by_agent(filtered))

    render_time_charts(filtered)

    st.subheader("Données détaillées")
    st.dataframe(filtered.head(500))

    export_data(filtered)


if __name__ == "__main__":
    main()
