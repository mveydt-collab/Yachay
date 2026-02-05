import os
import json
import requests
from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_PASSWORD = os.getenv("YACHAY_PASSWORD")  # <- set this in your .env

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# =========================
# Simple password gate
# =========================

def check_password() -> bool:
    """Simple single-password protection using session_state and env var."""
    if not APP_PASSWORD:
        st.error("App password not configured. Set YACHAY in your .env file.")
        return False

    # Already logged in
    if st.session_state.get("password_correct", False):
        return True

    def password_entered():
        """Callback when the user submits the password."""
        if st.session_state.get("password_input") == APP_PASSWORD:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    st.title("🧠 Yachay – Secure Access")
    st.text_input(
        "Enter password",
        type="password",
        key="password_input",
        on_change=password_entered,
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password. Try again.")

    return st.session_state.get("password_correct", False)

if not check_password():
    st.stop()

# ========= Everything below this is your existing app =========

def call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API failed: {resp.status_code}, {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def build_full_forecast_prompt(df: pd.DataFrame, user_instruction: str, months: int) -> str:
    schema_info = {
        "shape": [len(df), len(df.columns)],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(8).to_dict(orient="records"),
        "summary_stats": {
            col: {
                "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            }
            for col in df.select_dtypes(include=[np.number]).columns
        },
    }

    prompt = {
        "user_instruction": user_instruction,
        "data_schema": schema_info,
        "forecast_requirements": {
            "horizon_months": months,
            "output": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "forecast_table": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "month": {"type": "integer"},
                                "income": {"type": "number"},
                                "total_expenses": {"type": "number"},
                                "net_cashflow": {"type": "number"},
                                "cumulative_balance": {"type": "number"},
                                "key_breakdown": {
                                    "type": "object",
                                    "properties": {
                                        "fixed_expenses": {"type": "number"},
                                        "variable_expenses": {"type": "number"},
                                        "one_time_expenses": {"type": "number"},
                                    },
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["month", "income", "total_expenses", "net_cashflow", "cumulative_balance"],
                        },
                    },
                    "starting_balance": {"type": "number", "default": 0},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["analysis", "forecast_table", "starting_balance", "confidence"],
            },
        },
    }
    return json.dumps(prompt, indent=2)

st.title("🧠 Yachay – AI-Driven Financial Forecasting")

# Forecast horizon
st.header("Forecast Settings")
col1, col2 = st.columns([1, 3])
months = col1.number_input("Forecast months", min_value=1, max_value=60, value=12, step=1)
st.info(f"Generating {months}-month forecast")

# Mode selection
if "mode" not in st.session_state:
    st.session_state.mode = None

st.header("Choose your input method")
col1, col2 = st.columns(2)

if st.session_state.mode != "manual":
    if col1.button("📝 Manual Entry", use_container_width=True):
        st.session_state.mode = "manual"

if st.session_state.mode != "file":
    if col2.button("📁 Upload File (Fully AI-Analyzed)", use_container_width=True):
        st.session_state.mode = "file"

if st.session_state.mode:
    st.success(
        f"✅ **{st.session_state.mode.replace('manual', 'Manual Entry').replace('file', 'File Upload + AI')}** mode active"
    )
    st.markdown("---")
else:
    st.info("👆 Click a button above to get started")
    st.stop()

mode = st.session_state.mode
df_forecast = None

# Manual mode
if mode == "manual":
    st.subheader("📝 Enter your financial details")
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=4000.0, step=100.0)
        monthly_savings = st.number_input("Monthly Savings ($)", min_value=0.0, value=500.0, step=50.0)
        starting_balance = st.number_input("Starting Balance ($)", min_value=0.0, value=10000.0, step=100.0)
    with col2:
        rent = st.number_input("Rent / Mortgage ($)", min_value=0.0, value=1200.0, step=50.0)
        utilities = st.number_input("Utilities ($)", min_value=0.0, value=300.0, step=10.0)
        loan_payment = st.number_input("Loan Payments ($)", min_value=0.0, value=400.0, step=50.0)

    food_entertainment = st.number_input("Food & Entertainment ($)", min_value=0.0, value=600.0, step=50.0)
    transport = st.number_input("Transport ($)", min_value=0.0, value=200.0, step=20.0)

    with st.expander("Optional: Custom scenarios"):
        income_changes = {}
        one_time_expenses = {}
        col1, col2 = st.columns(2)
        num_inc = col1.number_input("Income changes", min_value=0, max_value=months, value=0, step=1)
        num_exp = col2.number_input("One-time expenses", min_value=0, max_value=months, value=0, step=1)

        for i in range(num_inc):
            cols = st.columns(2)
            m = cols[0].number_input(f"Inc Month {i+1}", 1, months, 1, key=f"man_m_inc_{i}")
            income_changes[m] = cols[1].number_input(f"Inc ${i+1}", 0.0, key=f"man_inc_{i}")

        for i in range(num_exp):
            cols = st.columns(2)
            m = cols[0].number_input(f"Exp Month {i+1}", 1, months, 1, key=f"man_m_exp_{i}")
            one_time_expenses[m] = cols[1].number_input(f"Exp ${i+1}", 0.0, key=f"man_exp_{i}")

    total_fixed = rent + utilities + loan_payment
    var_total = food_entertainment + transport
    monthly_expenses_base = total_fixed + var_total + monthly_savings

    current_balance = starting_balance
    forecast_data = []
    for m in range(1, months + 1):
        inc = income_changes.get(m, monthly_income)
        var_exp = var_total
        ot_exp = one_time_expenses.get(m, 0)
        total_expenses = total_fixed + var_exp + monthly_savings + ot_exp
        net_cashflow = inc - total_expenses
        current_balance += net_cashflow
        forecast_data.append(
            {
                "month": m,
                "income": inc,
                "total_expenses": total_expenses,
                "net_cashflow": net_cashflow,
                "cumulative_balance": current_balance,
            }
        )

    df_forecast = pd.DataFrame(forecast_data)

# File + AI mode
elif mode == "file":
    st.subheader("📁 Upload your financial data")
    uploaded_file = st.file_uploader("Upload Excel/CSV (any format works!)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("**Tell the AI about your data (optional):**")
            user_instruction = st.text_area(
                "",
                value=(
                    f"This is my financial data. Automatically detect income, expenses, "
                    f"categories, time periods, and create a realistic {months}-month forecast with trends."
                ),
                height=80,
            )

            if st.button("🧠 Generate Complete AI Forecast", type="primary"):
                with st.spinner("🤖 AI analyzing your data structure..."):
                    system_prompt = f"""You are an expert financial analyst. Analyze ANY financial dataset and output a COMPLETE {months}-month forecast.

CRITICAL RULES:
1. Your response MUST be 100% valid JSON - NO other text
2. Handle ANY schema: transactions, monthly summaries, categories, etc.
3. If no time column exists, distribute data proportionally across {months} months
4. Use historical patterns/trends for future months
5. Cumulative balance must accumulate correctly from starting_balance"""

                    user_prompt = build_full_forecast_prompt(df, user_instruction, months)
                    ai_forecast = call_groq_json(system_prompt, user_prompt)

                    st.success("🎉 AI Forecast Complete!")
                    st.markdown(f"**Confidence:** {ai_forecast['confidence'].upper()}")
                    st.markdown(f"**AI Analysis:** {ai_forecast['analysis']}")

                    df_forecast = pd.DataFrame(ai_forecast["forecast_table"])

        except Exception as e:
            st.error(f"❌ File error: {e}")

# Results
if df_forecast is not None:
    st.header(f"📊 {months}-Month Financial Forecast")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(
        df_forecast["month"],
        df_forecast["cumulative_balance"],
        marker="o",
        linewidth=4,
        markersize=10,
        color="#2E86AB",
        label="Balance",
    )
    ax1.fill_between(
        df_forecast["month"], df_forecast["cumulative_balance"], alpha=0.2, color="#2E86AB"
    )
    ax1.set_title("Cumulative Savings Balance", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Month")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Balance ($)")

    colors = ["#27AE60" if x >= 0 else "#E74C3C" for x in df_forecast["net_cashflow"]]
    ax2.bar(df_forecast["month"], df_forecast["net_cashflow"], color=colors, alpha=0.8, width=0.6)
    ax2.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax2.set_title("Monthly Net Cashflow", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Month")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Net Cashflow ($)")

    plt.tight_layout()
    st.pyplot(fig)

    col1, col2, col3, col4 = st.columns(4)
    total_income = df_forecast["income"].sum()
    total_expenses = df_forecast["total_expenses"].sum()
    final_balance = df_forecast["cumulative_balance"].iloc[-1]
    avg_net = df_forecast["net_cashflow"].mean()

    with col1:
        st.metric("💰 Total Income", f"${total_income:,.0f}")
    with col2:
        st.metric("💸 Total Expenses", f"${total_expenses:,.0f}")
    with col3:
        st.metric("🏦 Final Balance", f"${final_balance:,.0f}")
    with col4:
        st.metric("📈 Avg Monthly Net", f"${avg_net:.0f}")

    st.subheader("Detailed Monthly Breakdown")
    st.dataframe(df_forecast.round(0), use_container_width=True)

    st.subheader("💾 Download Results")

    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
        df_forecast.round(0).to_excel(writer, index=False, sheet_name="Forecast")
    st.download_button(
        label="📊 Excel",
        data=excel_data.getvalue(),
        file_name=f"yachay_forecast_{months}mo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_data = df_forecast.round(0).to_csv(index=False).encode()
    st.download_button(
        "📄 CSV",
        csv_data,
        f"yachay_forecast_{months}mo.csv",
        "text/csv",
    )
