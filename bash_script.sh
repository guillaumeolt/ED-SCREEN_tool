#!/bin/bash

# Define API URL
API_URL="http://127.0.0.1:5000/predict"

# Define JSON payload
JSON_PAYLOAD='{"prediction_type": "zscore","data": "data/data_exemple.csv"}'

# Send POST request
curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD"

# Define JSON payload
JSON_PAYLOAD='{"prediction_type": "smiles", "similarity_data": {"MCF7": "data/VEGA_similarity_exemple_MCF7.csv", "A549": "data/VEGA_similarity_exemple_A549.csv"}}'

# Send POST request
curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD"