
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Kwara Small Business Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# PRODUCT NAMES
# --------------------------------

product_names = {
    1: "Rice",
    2: "Sugar",
    3: "Beans",
    4: "Garri",
    5: "Noodles",
    6: "Vegetable Oil",
    7: "Semovita",
    8: "Flour",
    9: "Spaghetti",
    10: "Milk",
    11: "Tomato Paste",
    12: "Salt",
    13: "Bread",
    14: "Indomie",
    15: "Detergent"
}

# --------------------------------
# WEBSITE HEADING
# --------------------------------

st.title("📊 Kwara State Small Business Sales Forecasting System")

st.write(
    "A sales analysis and forecasting platform designed "
    "to help small businesses in Kwara State understand "
    "their sales performance and plan for future demand."
)

st.divider()

# --------------------------------
# LOAD DATASET
# --------------------------------

data_path = Path("data/train.csv")

if data_path.exists():

    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "sales", "store", "item"]
    )

    df["item"] = pd.to_numeric(
        df["item"],
        errors="coerce"
    )

    df = df.dropna(subset=["item"])
    df["item"] = df["item"].astype(int)

    st.success("Sales dataset loaded successfully!")

    # --------------------------------
    # SUMMARY INFORMATION
    # --------------------------------

    st.subheader("Business Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Sales Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Total Units Sold",
            f"{df['sales'].sum():,.0f}"
        )

    with col3:
        st.metric(
            "Number of Businesses",
            f"{df['store'].nunique():,}"
        )

    st.divider()

    # --------------------------------
    # EXPLORE SALES DATA
    # --------------------------------

    st.subheader("Explore Sales Data")

    col1, col2 = st.columns(2)

    with col1:

        products = sorted(df["item"].unique())

        selected_product = st.selectbox(
            "Select Product",
            products,
            format_func=lambda x: product_names.get(
                x, f"Product {x}"
            )
        )

    with col2:

        stores = sorted(df["store"].unique())

        selected_store = st.selectbox(
            "Select Business",
            stores
        )

    # --------------------------------
    # YEAR SELECTION
    # --------------------------------

    years = sorted(
        df["date"].dt.year.unique()
    )

    selected_years = st.multiselect(
        "Select Years to View",
        years,
        default=years
    )

    # --------------------------------
    # FILTER SALES DATA
    # --------------------------------

    filtered = df[
        (df["item"] == selected_product) &
        (df["store"] == selected_store) &
        (df["date"].dt.year.isin(selected_years))
    ].sort_values("date")

    st.divider()

    # --------------------------------
    # SELECTED BUSINESS INFORMATION
    # --------------------------------

    st.subheader("Selected Business Information")

    st.write(f"**Business:** {selected_store}")

    st.write(
        f"**Product:** "
        f"{product_names.get(selected_product, 'Unknown Product')}"
    )

    st.write(
        f"**Years Selected:** "
        f"{', '.join(map(str, selected_years)) if selected_years else 'None'}"
    )

    # --------------------------------
    # HISTORICAL SALES GRAPH
    # --------------------------------

    st.subheader("Historical Sales Analysis")

    if not filtered.empty:

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            filtered["date"],
            filtered["sales"],
            linewidth=1.2,
            label="Historical Sales"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")

        ax.set_title(
            f"{product_names.get(selected_product)} Sales "
            f"at {selected_store}"
        )

        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # --------------------------------
        # FILTERED SALES SUMMARY
        # --------------------------------

        st.subheader("Selected Sales Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Units Sold",
                f"{filtered['sales'].sum():,.0f}"
            )

        with col2:
            st.metric(
                "Average Daily Sales",
                f"{filtered['sales'].mean():,.2f}"
            )

        with col3:
            st.metric(
                "Highest Daily Sales",
                f"{filtered['sales'].max():,.0f}"
            )

        # --------------------------------
        # SALES RECORDS
        # --------------------------------

        st.subheader("Sales Records")

        display_data = filtered.copy()

        display_data["item"] = display_data["item"].map(
            lambda x: product_names.get(x, f"Product {x}")
        )

        st.dataframe(
            display_data.tail(20),
            use_container_width=True
        )

        # --------------------------------
        # DOWNLOAD DATA
        # --------------------------------

        st.download_button(
            "Download Filtered Sales Data",
            data=filtered.to_csv(index=False),
            file_name="kwara_filtered_sales.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No sales records found for your selected options."
        )

    # --------------------------------
    # FOOTER
    # --------------------------------

    st.divider()

    st.caption(
        "Kwara State Small Business Sales Forecasting System | "
        "Academic Project Demonstration | 2019–2025"
    )

else:

    st.error(
        "Dataset not found. Make sure data/train.csv "
        "exists inside your project folder."
    )
