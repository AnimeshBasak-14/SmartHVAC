#!/usr/bin/env bash
# run_pipeline.sh
set -e
echo "============================================================"
echo "  HVAC & Smart Appliance GML Pipeline (V100 GPU + Neo4j)    "
echo "============================================================"
nvidia-smi || echo "Warning: nvidia-smi not found, falling back to CPU/available device"
python3 train_and_monitor.py
echo "Training complete. Launching Graphical Dashboard on port 8501..."
streamlit run dashboard_app.py --server.port 8501 --server.headless true

