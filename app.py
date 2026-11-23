
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=["date", "sales"])

    return df

df = load_data()

# ==========================================
# HEADER
# ==========================================
st.title("📊 Sales Forecasting System")

st.write(
    "Analyse historical sales performance and estimate "
    "future sales to support business planning."
)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Dashboard Filters")

# Store filter
stores = ["All Stores"] + sorted(df["store"].unique().tolist())

selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)

# Product filter
items = ["All Products"] + sorted(df["item"].unique().tolist())

selected_item = st.sidebar.selectbox(
    "Select Product",
    items
)

# Year filter
available_years = sorted(df["date"].dt.year.unique().tolist())

selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    available_years,
    default=available_years
)

# ==========================================
# FILTER DATA
# ==========================================
filtered_df = df.copy()

if selected_store != "All Stores":
    filtered_df = filtered_df[
        filtered_df["store"] == selected_store
    ]

if selected_item != "All Products":
    filtered_df = filtered_df[
        filtered_df["item"] == selected_item
    ]

filtered_df = filtered_df[
    filtered_df["date"].dt.year.isin(selected_years)
]

# ==========================================
# METRICS
# ==========================================
st.subheader("Sales Overview")

total_records = len(filtered_df)
total_sales = filtered_df["sales"].sum()
total_stores = filtered_df["store"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Total Sales", f"{total_sales:,.0f}")
col3.metric("Stores Selected", f"{total_stores:,}")

st.divider()

# ==========================================
# MONTHLY HISTORICAL SALES
# ==========================================
st.subheader("Historical Monthly Sales")

if not filtered_df.empty:

    monthly_sales = (
        filtered_df.set_index("date")
        .resample("MS")["sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(
        monthly_sales.index,
        monthly_sales.values,
        width=20,
        label="Historical Sales"
    )

    ax.set_title("Monthly Sales Performance")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("No sales records found.")
    st.stop()

st.divider()

# ==========================================
# SALES SUMMARY
# ==========================================
st.subheader("Sales Summary")

daily_sales = filtered_df.groupby("date")["sales"].sum()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"{daily_sales.sum():,.0f}"
)

col2.metric(
    "Average Daily Sales",
    f"{daily_sales.mean():,.2f}"
)

col3.metric(
    "Highest Daily Sales",
    f"{daily_sales.max():,.2f}"
)

st.divider()

# ==========================================
# FUTURE SALES FORECAST
# ==========================================
st.subheader("🔮 Future Sales Forecast")

st.write(
    "The system uses historical monthly sales patterns "
    "to estimate sales for the next six months."
)

if len(monthly_sales) >= 6:

    # Prepare training data
    forecast_data = monthly_sales.reset_index()
    forecast_data.columns = ["date", "sales"]

    forecast_data["time_index"] = np.arange(
        len(forecast_data)
    )

    # Seasonal features
    forecast_data["month"] = forecast_data["date"].dt.month

    forecast_data["sin_month"] = np.sin(
        2 * np.pi * forecast_data["month"] / 12
    )

    forecast_data["cos_month"] = np.cos(
        2 * np.pi * forecast_data["month"] / 12
    )

    X = forecast_data[
        ["time_index", "sin_month", "cos_month"]
    ]

    y = forecast_data["sales"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Create next 6 months
    future_dates = pd.date_range(
        start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS"
    )

    future_df = pd.DataFrame({
        "date": future_dates
    })

    future_df["time_index"] = np.arange(
        len(forecast_data),
        len(forecast_data) + 6
    )

    future_df["month"] = future_df["date"].dt.month

    future_df["sin_month"] = np.sin(
        2 * np.pi * future_df["month"] / 12
    )

    future_df["cos_month"] = np.cos(
        2 * np.pi * future_df["month"] / 12
    )

    future_X = future_df[
        ["time_index", "sin_month", "cos_month"]
    ]

    # Generate predictions
    future_df["forecast_sales"] = model.predict(future_X)

    # Avoid negative forecasts
    future_df["forecast_sales"] = future_df[
        "forecast_sales"
    ].clip(lower=0)

    # ==========================================
    # FORECAST METRICS
    # ==========================================
    total_forecast = future_df["forecast_sales"].sum()
    average_forecast = future_df["forecast_sales"].mean()

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Sales (Next 6 Months)",
        f"{total_forecast:,.0f}"
    )

    col2.metric(
        "Average Monthly Forecast",
        f"{average_forecast:,.0f}"
    )

    # ==========================================
    # FORECAST CHART
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical bars
    ax.bar(
        monthly_sales.index,
        monthly_sales.values,
        width=20,
        label="Historical Sales"
    )

    # Forecast bars
    ax.bar(
        future_df["date"],
        future_df["forecast_sales"],
        width=20,
        label="Forecast Sales"
    )

    ax.set_title("Historical Sales and 6-Month Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # ==========================================
    # FORECAST TABLE
    # ==========================================
    st.subheader("Forecast Results")

    display_forecast = future_df[
        ["date", "forecast_sales"]
    ].copy()

    display_forecast["date"] = (
        display_forecast["date"].dt.strftime("%B %Y")
    )

    display_forecast.columns = [
        "Forecast Month",
        "Predicted Sales"
    ]

    display_forecast["Predicted Sales"] = (
        display_forecast["Predicted Sales"].round(2)
    )

    st.dataframe(
        display_forecast,
        width="stretch",
        hide_index=True
    )

    # Forecast CSV download
    forecast_csv = future_df[
        ["date", "forecast_sales"]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast Results",
        data=forecast_csv,
        file_name="six_month_sales_forecast.csv",
        mime="text/csv"
    )

    st.caption(
        "Forecasts are model-based estimates and are not guaranteed "
        "to represent actual future sales."
    )

else:
    st.warning(
        "At least six months of historical sales data are required "
        "to generate a forecast."
    )

st.divider()

# ==========================================
# SALES RECORDS
# ==========================================
st.subheader("Sales Records")

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True
)

# ==========================================
# DOWNLOAD FILTERED SALES
# ==========================================
csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Sales Data as CSV",
    data=csv_data,
    file_name="sales_records.csv",
    mime="text/csv"
)

# ==========================================
# FOOTER
# ==========================================
st.caption(
    "Sales Forecasting System | Python | Pandas | "
    "Scikit-learn | Streamlit"
)
