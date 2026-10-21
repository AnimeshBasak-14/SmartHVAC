@echo off
echo ============================================================
echo   HVAC & Smart Appliance GML Pipeline (Windows Local / Lab)
echo ============================================================
py -3.11 train_and_monitor.py
echo Training complete. Launching Graphical Dashboard on port 8501...
py -3.11 -m streamlit run dashboard_app.py --server.port 8501
pause

