
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
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

    return df

df = load_data()

# ==========================================
# TITLE
# ==========================================
st.title("Sales Forecasting System")

st.write(
    "An interactive dashboard for analysing sales performance, "
    "stores, products, and sales trends."
)

# ==========================================
# SIDEBAR FILTERS
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
# DASHBOARD METRICS
# ==========================================
st.subheader("Sales Overview")

total_records = len(filtered_df)
total_sales = filtered_df["sales"].sum()
total_stores = filtered_df["store"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Total Sales", f"{total_sales:,.0f}")
col3.metric("Stores", f"{total_stores:,}")

st.divider()

# ==========================================
# MONTHLY BAR CHART
# ==========================================
st.subheader("Sales Performance")

if not filtered_df.empty:

    monthly_sales = (
        filtered_df.set_index("date")
        .resample("MS")["sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create actual monthly bars
    ax.bar(
        range(len(monthly_sales)),
        monthly_sales.values,
        width=0.75
    )

    ax.set_title("Monthly Sales Performance")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Show one label every three months
    tick_positions = range(0, len(monthly_sales), 3)

    ax.set_xticks(list(tick_positions))

    ax.set_xticklabels(
        monthly_sales.index.strftime("%b %Y")[::3],
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

else:
    st.warning("No sales data available for the selected filters.")

st.divider()

# ==========================================
# SALES SUMMARY
# ==========================================
st.subheader("Sales Summary")

if not filtered_df.empty:

    daily_sales = filtered_df.groupby("date")["sales"].sum()

    total_sales_value = daily_sales.sum()
    average_daily_sales = daily_sales.mean()
    highest_daily_sales = daily_sales.max()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"{total_sales_value:,.0f}"
    )

    col2.metric(
        "Average Daily Sales",
        f"{average_daily_sales:,.2f}"
    )

    col3.metric(
        "Highest Daily Sales",
        f"{highest_daily_sales:,.2f}"
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
# DOWNLOAD CSV
# ==========================================
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

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
    "Sales Forecasting System | Developed with Python and Streamlit"
)
