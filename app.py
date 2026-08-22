import streamlit as st

st.set_page_config(page_title="Pokémon Battle Simulator", page_icon="⚔️", layout="wide")

pages = [
    st.Page("arena.py", title="Battle the boss", icon="⚔️", default=True),
    st.Page("sandbox.py", title="Sandbox", icon="🛠️"),
]
st.navigation(pages).run()
