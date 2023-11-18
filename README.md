streamlit run Home.py

gunicorn -w 4 -k uvicorn.workers.UvicornWorker segment_api:app

