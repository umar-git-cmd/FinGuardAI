import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
API_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Основные стили карточек и метрик */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Стильные бейджи статуса */
    .badge-fraud {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .badge-legit {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 🧠 Session State Management
# ==========================================
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_analyzed": 0,
        "fraud_detected": 0,
    }


# ==========================================
# 🌐 API Client Functions
# ==========================================
def check_api_health() -> Tuple[bool, Optional[str]]:
    """Проверка доступности API."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, None
        return False, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException:
        return False, "Cannot connect to Fraud Detection API. Make sure the FastAPI server is running."


def get_model_info() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Загрузка конфигурации модели и списка признаков."""
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Error {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException:
        return None, "Unable to load model metadata from backend."


def predict_transaction(tx_data: Dict[str, float]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Проверка одной транзакции."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=tx_data, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return resp.json(), None
        detail = resp.json().get("detail", resp.text) if resp.headers.get(
            "content-type") == "application/json" else resp.text
        return None, f"Analysis error ({resp.status_code}): {detail}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The server took too long to respond."
    except requests.exceptions.RequestException:
        return None, "Failed to send transaction to API. Please check your connection."


def predict_csv(file_bytes: bytes, filename: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Пакетная проверка транзакций через отправку CSV файла."""
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        resp = requests.post(f"{API_URL}/predict/csv", files=files, timeout=TIMEOUT_SECONDS * 5)
        if resp.status_code == 200:
            return resp.json(), None

        detail = resp.text
        try:
            parsed = resp.json()
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = parsed["detail"]
        except Exception:
            pass
        return None, f"Batch check failed ({resp.status_code}): {detail}"
    except requests.exceptions.Timeout:
        return None, "CSV analysis timed out. The dataset may be too large."
    except requests.exceptions.RequestException:
        return None, "Cannot connect to Fraud Detection API during CSV upload."


# ==========================================
# 📊 UI Pages
# ==========================================
def render_dashboard(is_online: bool, model_data: Optional[Dict[str, Any]]):
    st.title("🛡️ Fraud Detection System")
    st.caption("AI-powered transaction fraud detection")

    total = st.session_state.session_stats["total_analyzed"]
    frauds = st.session_state.session_stats["fraud_detected"]
    fraud_rate = (frauds / total * 100) if total > 0 else 0.0

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_text = "🟢 Online" if is_online else "🔴 Offline"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">API Status</div>
                <div class="metric-value" style="font-size: 1.5rem;">{status_text}</div>
                <div class="metric-sub">{API_URL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Processed in Session</div>
                <div class="metric-value">{total:,}</div>
                <div class="metric-sub">Total verified transactions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Detected Frauds</div>
                <div class="metric-value" style="color: #f87171;">{frauds:,}</div>
                <div class="metric-sub">Threats blocked or flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Suspicious Ratio</div>
                <div class="metric-value" style="color: {'#fbbf24' if fraud_rate > 5 else '#60a5fa'};">{fraud_rate:.2f}%</div>
                <div class="metric-sub">Fraudulent share of volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    st.subheader("Quick Overview & Guidelines")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            """
            **How to use this system:**
            - **Single Transaction:** Check an individual transaction quickly by inputting the feature values manually.
            - **CSV Analysis:** Upload banking registries or batch logs in `.csv` format to get bulk security insights.
            - **Model Information:** Inspect the active classifier, operational thresholds, and parameters.
            """
        )
    with c_right:
        if model_data:
            st.info(
                f"Active Model: **{model_data.get('model', 'Unknown')}**  \n"
                f"Decision Threshold: **{model_data.get('threshold', 0.5):.4f}**  \n"
                f"Features required: **{model_data.get('n_features', 0)}**"
            )
        else:
            st.warning("Backend API is currently offline. Please ensure the FastAPI server is running.")


def render_single_transaction(model_data: Optional[Dict[str, Any]]):
    st.title("Single Transaction Inspection")
    st.caption("Perform immediate real-time risk scoring on an incoming transaction.")

    if not model_data:
        st.error("Cannot load feature inputs because the backend API is unreachable.")
        return

    features: List[str] = model_data.get("features", [])
    threshold: float = float(model_data.get("threshold", 0.5))

    st.write("")
    st.subheader("Transaction Parameters")
    st.caption("Fields are dynamically loaded from model specifications.")

    # Динамическое распределение полей ввода по колонкам (до 3 колонок)
    cols = st.columns(3)
    user_inputs = {}

    for idx, feat in enumerate(features):
        col = cols[idx % 3]
        with col:
            user_inputs[feat] = st.number_input(
                label=f"📌 {feat}",
                value=0.0,
                format="%.5f",
                help=f"Normalized or raw numerical value for feature '{feat}'",
                key=f"input_{feat}",
            )

    st.write("")
    analyze_btn = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Analyzing risk probability with AI model..."):
            result, err = predict_transaction(user_inputs)

        if err:
            st.error(f"⚠️ {err}")
            return

        prob = result["probability"]
        is_fraud = result["is_fraud"]
        thresh = result["threshold"]

        # Обновляем статистику сессии
        st.session_state.session_stats["total_analyzed"] += 1
        if is_fraud:
            st.session_state.session_stats["fraud_detected"] += 1

        st.write("")
        st.divider()

        # Крупное понятное визуальное заключение
        if is_fraud:
            st.markdown(
                f"""
                <div class="badge-fraud">
                    <span>🚨</span>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700;">Potential Fraud Detected</div>
                        <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">High risk indicators present. Transaction flagged for manual review or decline.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="badge-legit">
                    <span>✅</span>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700;">Transaction Appears Legitimate</div>
                        <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">Low risk profile. Probability is within normal boundaries.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Probability", f"{prob * 100:.2f}%")
        m2.metric("Safety Threshold", f"{thresh * 100:.2f}%")
        m3.metric("Verdict", "FRAUD" if is_fraud else "LEGITIMATE")

        st.write("**Probability Meter**")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Model threshold is set at **{thresh * 100:.2f}%**. "
            f"The transaction is marked as fraud if probability meets or exceeds this limit."
        )


def render_csv_analysis(model_data: Optional[Dict[str, Any]]):
    st.title("Batch CSV Transaction Analysis")
    st.caption("Upload transactional datasets for bulk scam and anomaly classification.")

    if not model_data:
        st.error("API backend is unreachable. Mass analysis is currently disabled.")
        return

    features = model_data.get("features", [])
    max_batch = model_data.get("max_batch", 10000)

    uploaded_file = st.file_uploader("Select a CSV file for verification", type=["csv"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # Валидация чтения на стороне клиента для быстрого отклика
        try:
            df_preview = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            st.error("Invalid file: The file could not be parsed as a valid CSV.")
            return

        if df_preview.empty:
            st.error("The uploaded CSV file is empty.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("File Name", uploaded_file.name)
        c2.metric("Rows Count", f"{len(df_preview):,}")
        c3.metric("Columns", len(df_preview.columns))

        # Быстрая проверка колонок до отправки на бэк
        missing = [f for f in features if f not in df_preview.columns]
        if missing:
            st.warning(
                f"⚠️ Warning: Missing required model features in CSV: {missing[:5]} (Total missing: {len(missing)})")

        with st.expander("👀 Preview First 5 Rows of Uploaded File", expanded=True):
            st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("🚀 Analyze CSV", type="primary", use_container_width=True):
            if len(df_preview) > max_batch:
                st.error(f"Dataset exceeds the maximum batch limit of {max_batch:,} rows.")
                return

            with st.spinner("Processing batch on Fraud Detection Engine..."):
                batch_response, error_msg = predict_csv(file_bytes, uploaded_file.name)

            if error_msg:
                st.error(f"⚠️ {error_msg}")
                return

            total_count = batch_response.get("count", 0)
            fraud_count = batch_response.get("fraud_count", 0)
            legit_count = total_count - fraud_count
            fraud_pct = (fraud_count / total_count * 100) if total_count > 0 else 0.0

            # Синхронизация со статистикой сессии
            st.session_state.session_stats["total_analyzed"] += total_count
            st.session_state.session_stats["fraud_detected"] += fraud_count

            st.success(f"Batch analysis completed successfully: {total_count:,} records processed.")

            # Метрики
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{total_count:,}")
            m2.metric("Fraud Count", f"{fraud_count:,}")
            m3.metric("Legitimate Count", f"{legit_count:,}")
            m4.metric("Batch Fraud Rate", f"{fraud_pct:.2f}%")

            # Подготовка таблицы с результатами
            results = batch_response.get("results", [])
            df_results = df_preview.copy()
            df_results["Fraud_Probability"] = [r["probability"] for r in results]
            df_results["Status"] = ["🔴 FRAUD" if r["is_fraud"] else "🟢 LEGITIMATE" for r in results]

            st.write("")
            st.subheader("Data Visualizations")

            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                st.write("**Verdict Breakdown**")
                chart_data = pd.DataFrame(
                    {
                        "Class": ["Legitimate", "Fraud"],
                        "Count": [legit_count, fraud_count],
                    }
                ).set_index("Class")
                st.bar_chart(chart_data)

            with ch_col2:
                st.write("**Probability Distribution**")
                # Разбивка на 10 корзин
                bins = pd.cut(df_results["Fraud_Probability"], bins=10).value_counts().sort_index()
                dist_df = pd.DataFrame({"Range": [str(b) for b in bins.index], "Count": bins.values}).set_index("Range")
                st.bar_chart(dist_df)

            st.write("")
            st.subheader("Detailed Evaluation Table")
            st.dataframe(
                df_results,
                use_container_width=True,
            )

            # Кнопка скачивания результатов
            csv_output = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated CSV",
                data=csv_output,
                file_name=f"inspected_{uploaded_file.name}",
                mime="text/csv",
            )


def render_model_info(model_data: Optional[Dict[str, Any]]):
    st.title("Model Architecture & Parameters")
    st.caption("Detailed specification of the deployed machine learning model.")

    if not model_data:
        st.error("Cannot connect to Fraud Detection API. Make sure the FastAPI server is running.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", model_data.get("model", "N/A"))
    c2.metric("Features Count", model_data.get("n_features", 0))
    c3.metric("Decision Threshold", f"{model_data.get('threshold', 0):.4f}")
    c4.metric("Max Batch Size", f"{model_data.get('max_batch', 0):,}")

    st.write("")
    st.info(
        "💡 **Threshold** — значение вероятности, выше которого транзакция считается потенциально мошеннической. "
        "Оно откалибровано для минимизации финансовых потерь при контроле ложных срабатываний."
    )

    st.write("")
    st.subheader("Required Features Schema")
    features = model_data.get("features", [])

    with st.expander(f"List of Input Features ({len(features)})", expanded=True):
        f_cols = st.columns(3)
        for i, feat in enumerate(features):
            f_cols[i % 3].code(feat, language="text")


def render_api_status(is_online: bool, error_msg: Optional[str]):
    st.title("API Gateway Health & Status")
    st.caption("Monitoring connection to the core machine learning inference service.")

    st.write("")
    if is_online:
        st.success("🟢 **API Online** — Inference engine is operational and serving requests.")
        st.markdown(
            f"""
            - **Endpoint:** `{API_URL}`
            - **Health Check:** `/health`
            - **Protocol:** HTTP/1.1 REST
            """
        )
    else:
        st.error("🔴 **API Offline**")
        st.warning("Cannot connect to Fraud Detection API. Make sure the FastAPI server is running.")
        if error_msg:
            with st.expander("Technical details"):
                st.code(error_msg, language="bash")


# ==========================================
# 🧭 Sidebar Navigation & App Entrypoint
# ==========================================
def main():
    # Проверка статуса и загрузка метаданных модели
    is_online, health_err = check_api_health()
    model_data, _ = get_model_info() if is_online else (None, None)

    with st.sidebar:
        st.markdown("### 🛡️ Fraud System")
        st.caption("AI Security Operations")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Single Transaction",
                "CSV Analysis",
                "Model Information",
                "API Status",
            ],
            index=0,
        )

        st.divider()
        st.caption("Backend Status")
        if is_online:
            st.markdown("🟢 **API Connected**")
        else:
            st.markdown("🔴 **API Disconnected**")

        st.divider()
        if st.button("Reset Session Counters", use_container_width=True):
            st.session_state.session_stats = {"total_analyzed": 0, "fraud_detected": 0}
            st.rerun()

    # Роутинг страниц
    if page == "Dashboard":
        render_dashboard(is_online, model_data)
    elif page == "Single Transaction":
        render_single_transaction(model_data)
    elif page == "CSV Analysis":
        render_csv_analysis(model_data)
    elif page == "Model Information":
        render_model_info(model_data)
    elif page == "API Status":
        render_api_status(is_online, health_err)


if __name__ == "__main__":
    main()