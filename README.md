# Chemical Compound Prediction API

## Overview
This Flask-based API provides predictions for chemical compounds using both z-score data and SMILES-based similarity. The system supports predictions for multiple cell lines (MCF7, A549) and endpoints (ER, AR, TR), using pre-trained machine learning models.

## Features
- Dual prediction modes:
  - Z-score based predictions using direct compound data
  - SMILES-based predictions using structural similarity
- Support for multiple cell lines (MCF7, A549)
- Multiple endpoints (ER, AR, TR)
- Applicability domain assessment
- RESTful API interface
- Command-line configuration options

## Installation

### Prerequisites
- Python 3.9.7 or higher
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server
Basic usage with default settings:
```bash
python app.py
```

Custom configuration:
```bash
python app.py --data-dir /path/to/data --model-dir /path/to/models --port 8000 --debug
```

Available command-line arguments:
- `--data-dir`: Directory containing data files (default: './data')
- `--model-dir`: Directory containing model files (default: './model')
- `--host`: Host to run the server on (default: '0.0.0.0')
- `--port`: Port to run the server on (default: 5000)
- `--debug`: Run in debug mode (flag)

### API Endpoints

#### POST /predict
Makes predictions for chemical compounds using either z-score data or SMILES similarity.

##### Z-score Based Prediction
Request:
```json
{
    "prediction_type": "zscore",
    "data": "PATH TO Z-SCORE VALUES"
}
```

##### SMILES Based Prediction
Request:
```json
{
    "prediction_type": "smiles",
    "similarity_data": {
        "MCF7": "PATH TO VEGA SIMILARRITY AGAINST CHEMICALS OF mat_MCF7_24h_10uM_all_clean_desc_ER_AR_TR",
        "A549": "PATH TO VEGA SIMILARRITY AGAINST CHEMICALS OF mat_A549_24h_10uM_all_clean_desc_ER_AR_TR"
    }
}
```

Response Format:
```json
{
    "MCF7": {
        "predictions": {
            "ER": [0.75],
            "AR": [0.62],
            "TR": [0.45]
        },
        "applicability_domain": {
            "ADI_ER": [1],
            "ADI_AR": [1],
            "ADI_TR": [0]
        }
    },
    "A549": {
        ...
    }
}
```

##### Exemple bash usage
```
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
```

## License
[Insert License Information]

## Contact
[Insert Contact Information]