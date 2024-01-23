## Running the local website
streamlit run Home.py

## Backend API
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --reload --timeout 1000


## Convert to a docker use case
Todo: All


### To Train the model  

python3 train_model.py  
python3 train_autoencoder.py  