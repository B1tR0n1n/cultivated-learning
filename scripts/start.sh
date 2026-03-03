#!/bin/bash
# Launch Cultivated Learning UI and Jupyter notebook

BASE=/workspace/Projects/cultivated-learning-24b

# Jupyter in background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

# Gradio UI with warnings suppressed at interpreter level
python3 -W ignore "$BASE/ui/app.py"
