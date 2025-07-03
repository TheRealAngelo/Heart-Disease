import streamlit as st
import pandas as pd
import numpy as np

st.title("🩺 Heart Disease Predictor - Test")
st.write("If you can see this, Streamlit is working!")

# Test basic functionality
st.write("## Testing basic components:")

# Test input
name = st.text_input("Enter your name:", "Test User")
st.write(f"Hello, {name}!")

# Test sidebar
st.sidebar.write("Sidebar works!")

# Test columns
col1, col2 = st.columns(2)
with col1:
    st.write("Column 1")
with col2:
    st.write("Column 2")

# Test button
if st.button("Test Button"):
    st.success("Button works!")

st.write("✅ If you can see this, the basic app structure is working!")
