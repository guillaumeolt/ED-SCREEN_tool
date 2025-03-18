from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from pathlib import Path
from typing import Dict, Union, List

class PredictionAPI:
    def __init__(self, data_dir, model_dir):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.load_resources()

    def load_resources(self):
        # Load models and descriptors
        self.models = {}
        self.descriptors = {}
        self.database = {}
        self.similarity_vega = {}
        
        for cell_line in ['MCF7', 'A549']:
            self.models[cell_line] = {}
            self.descriptors[cell_line] = {}
            
            # Load data and similarity matrices
            self.database[cell_line] = pd.read_csv(
                self.data_dir / f"mat_{cell_line}_24h_10uM_all_clean_desc_ER_AR_TR.csv",
                sep="\t"
            )
            
            # Load models and descriptors for each endpoint
            for endpoint in ['ER', 'AR', 'TR']:
                model_type = 'rf' if endpoint == 'ER' else 'tpot'
                
                # Load descriptors
                desc_path = self.model_dir / f"list_desc_{endpoint}_{cell_line}.txt"
                self.descriptors[cell_line][endpoint] = pd.read_csv(desc_path, header=None)[0].astype(str).tolist()
                
                # Load model
                model_path = self.model_dir / f"model_{endpoint}_{cell_line}_{model_type}.sav"
                with open(model_path, 'rb') as f:
                    self.models[cell_line][endpoint] = pickle.load(f)

        # Load gene info
        self.gene_info = pd.read_csv(self.data_dir / "geneinfo_beta.txt", sep="\t")
        self.landmark_genes = self.gene_info[
            self.gene_info["feature_space"] == "landmark"
        ]["gene_id"].astype(str).tolist()

    def predict_zscore(self, data: pd.DataFrame, cell_line) -> Dict:
        """Make predictions using z-score data"""
        results = {}

        # Make predictions
        predictions = {}
        for endpoint, model in self.models[cell_line].items():
            features = self.descriptors[cell_line][endpoint]
            predictions[f'{cell_line}_{endpoint}'] = model.predict_proba(data[features])[:, 1].tolist()
        
        # Calculate applicability domain
        adi = self.calculate_adi(data, cell_line)
        
        results[cell_line] = {
            'predictions': predictions,
            'applicability_domain': adi
        }
            
        return results[cell_line]

    def predict_smiles(self, similarity_data: Dict[str, List[float]]) -> Dict:
        """Make predictions using SMILES similarity data"""
        results = {}
        k = 6  # number of nearest neighbors for kNN approach

        for cell_line in ['MCF7', 'A549']:
            sim_df = pd.read_csv(similarity_data[cell_line], sep="\t") # TODO allow passing list ?

            # Check if exact match exists
            exact_match = sim_df[sim_df['VEGA_similarity'] == 1]
            
            if not exact_match.empty:
                # Use exact match with sig_id
                id_sig_id = exact_match['sig_id'].values[0]
                compound_data = self.database[cell_line].loc[
                    self.database[cell_line]['sig_id'] == id_sig_id
                ]
                
                # Make predictions
                predictions = {}
                for endpoint, model in self.models[cell_line].items():
                    features = self.descriptors[cell_line][endpoint]
                    predictions[f'{cell_line}_{endpoint}'] = model.predict_proba(compound_data[features])[:, 1].tolist()
                
                # Calculate ADI
                adi = self.calculate_adi(compound_data, cell_line)
                
            else:
                # Use k-NN approach
                most_similar_ids = sim_df.nlargest(k, 'VEGA_similarity')['sig_id']
                similar_compounds = self.database[cell_line][
                    self.database[cell_line]['sig_id'].isin(most_similar_ids)
                ]

                # Make predictions
                predictions = {}
                for endpoint, model in self.models[cell_line].items():
                    features = self.descriptors[cell_line][endpoint]
                    pred_values = model.predict_proba(similar_compounds[features])[:, 1]
                    # Take mean of k nearest neighbors
                    predictions[f'{cell_line}_{endpoint}'] = [float(np.mean(pred_values))]
                # Fill ADI with out of ADI values
                adi = {
                    f'ADI_{cell_line}_ER': [0],
                    f'ADI_{cell_line}_AR': [0],
                    f'ADI_{cell_line}_TR': [0]
                }
            
            results[cell_line] = {
                'predictions': predictions,
                'applicability_domain': adi
            }
        
        return results

    def calculate_adi(self, data, cell_line, threshold=0.3):
        """Calculate applicability domain index"""
        adi = {}
        activities = {'ER': 'Activite_ER', 'AR': 'Activite_AR', 'TR': 'Activite_TR'}
        
        for endpoint, activity in activities.items():
            endpoint_relative_data = self.database[cell_line][~self.database[cell_line][activity].isna()]
            similarities = cosine_similarity(
                data[self.landmark_genes],
                endpoint_relative_data[self.landmark_genes]
            )
            #mat_cosine_similarity = pd.DataFrame(similarities, columns = endpoint_relative_data.index)
            mean_similarities = pd.DataFrame(similarities).apply(
                lambda x: np.mean(sorted(x, reverse=True)[:3]), axis=1
            )
            adi[f'ADI_{cell_line}_{endpoint}'] = (mean_similarities > threshold).astype(int).tolist()
            
        return adi

def create_app(data_dir, model_dir):
    app = Flask(__name__)
    predictor = PredictionAPI(data_dir, model_dir)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No input data provided"}), 400

            # Check prediction type
            if 'prediction_type' not in data:
                return jsonify({"error": "Prediction type not specified"}), 400

            if data['prediction_type'] == 'zscore':
                if 'data_MCF7' not in data and 'data_A549' not in data:
                    return jsonify({"error": "No z-score data provided for the MCF7 cell line and A549 cell line"}), 400
                results = {}
                if 'data_MCF7' in data:
                    input_df = pd.read_csv(data['data_MCF7'], sep="\t")
                    results["MCF7"] = predictor.predict_zscore(input_df, cell_line='MCF7')
                if 'data_A549' in data:
                    input_df = pd.read_csv(data['data_A549'], sep="\t")
                    results["A549"] = predictor.predict_zscore(input_df, cell_line='A549')
            
            elif data['prediction_type'] == 'smiles':
                if 'similarity_data' not in data:
                    return jsonify({"error": "No similarity data provided"}), 400
                results = predictor.predict_smiles(data['similarity_data']) # TODO pd.DataFrame(data['similarity_data'])
            
            else:
                return jsonify({"error": "Invalid prediction type"}), 400

            return jsonify(results), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

def main():
    parser = argparse.ArgumentParser(description='Start the prediction API server')
    parser.add_argument('--data-dir', type=str, default='./data',
                      help='Directory containing data files')
    parser.add_argument('--model-dir', type=str, default='./model',
                      help='Directory containing model files')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode')

    args = parser.parse_args()
    
    app = create_app(args.data_dir, args.model_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()