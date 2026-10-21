"""
app.py
Streamlit Cloud Entrypoint for HVAC & Smart Appliance GML Command Center.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "hvac_gml_project")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Run the command center dashboard
import dashboard_app

