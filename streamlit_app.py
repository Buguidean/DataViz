import streamlit as st
from app import final_dashboard

st.set_page_config(
	layout="wide"
)

st.title("Visa-Free Destinations: Global Trends and Changes")
st.altair_chart(final_dashboard, use_container_width=True)
