from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Example Python function
def process_input(data):
    return {"message": f"Processed data: {data}"}

def process_input_prediction(data):
    data = pd.DataFrame(data)
    predictions = make_predictions(data, models, descriptors)
    print(predictions)
    return {"message": f"Processed data: {predictions}"}

# Load descriptors
def load_descriptors(file_path):
    return pd.read_csv(file_path, header=None)[0].astype(str).tolist()

# Load models
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Make predictions
def make_predictions(data, models, descriptors):
    predictions = {}
    for key in models:
        predictions[key] = models[key].predict_proba(data[descriptors[key]])[:, 1].tolist()
    return predictions
# Compute the applicability domain index
def make_applicability_domain(data, database, list_genes, thresholds):
    adi_predictions = {}
    dic_endpoint_activite = {"ER":"Activite_ER", "AR":"Activite_AR", "TR":"Activite_TR"}
    for key, key_activite in dic_endpoint_activite.items():
        database_adi = database[~database[key_activite].isna()]
        mat_cosine_similarity = cosine_similarity(X = data[list_genes],
                                                   Y = database_adi[list_genes])
        mat_cosine_similarity = pd.DataFrame(mat_cosine_similarity, 
             columns = database_adi.index)#,columns = list_id_validation_PC3)
        mat_mean_3_cosine = mat_cosine_similarity.apply(lambda x: np.mean(sorted(x, reverse=True)[0:3]), axis=1)
        print((mat_mean_3_cosine > thresholds[key]).astype(int).tolist())
        adi_predictions["ADI_"+key] = (mat_mean_3_cosine > thresholds[key]).astype(int).tolist()
    return adi_predictions

descriptors = {
    "MCF7": {
        "ER": load_descriptors("model/list_desc_ER_MCF7.txt"),
        "AR": load_descriptors("model/list_desc_AR_MCF7.txt"),
        "TR": load_descriptors("model/list_desc_TR_MCF7.txt")
    },
    "A549": {
        "ER": load_descriptors("model/list_desc_ER_A549.txt"),
        "AR": load_descriptors("model/list_desc_AR_A549.txt"),
        "TR": load_descriptors("model/list_desc_TR_A549.txt")
    }
}
models = {
    "MCF7": {
        "ER": load_model("model/model_ER_MCF7_rf.sav"),
        "AR": load_model("model/model_AR_MCF7_tpot.sav"),
        "TR": load_model("model/model_TR_MCF7_tpot.sav")
    },
    "A549": {
        "ER": load_model("model/model_ER_A549_rf.sav"),
        "AR": load_model("model/model_AR_A549_tpot.sav"),
        "TR": load_model("model/model_TR_A549_tpot.sav")
    }
}
thresholds = {
    "MCF7": {
        "ER": 0.3,
        "AR": 0.3,
        "TR": 0.3
    },
    "A549": {
        "ER": 0.3,
        "AR": 0.3,
        "TR": 0.3
    }
}
database = {
    "MCF7": pd.read_csv("data/mat_MCF7_24h_10uM_all_clean_desc_ER_AR_TR.csv", sep="\t"),
    "A549": pd.read_csv("data/mat_A549_24h_10uM_all_clean_desc_ER_AR_TR.csv", sep="\t")
}

similarity_vega = {
    "MCF7": pd.read_csv("data/VEGA_similarity_exemple_MCF7.csv", sep="\t"),
    "A549": pd.read_csv("data/VEGA_similarity_exemple_A549.csv", sep="\t")
}
if True: # Prediction with z-score values
    # Load gene information and user data
    gene_infos = pd.read_csv("data/geneinfo_beta.txt", sep="\t")
    data = pd.read_csv("data/data_exemple.csv", sep="\t")
    predictions = {}
    for key in database:
        predictions[key] = make_predictions(data, models[key], descriptors[key])
        list_genes_landmark = list(gene_infos["gene_id"][gene_infos["feature_space"] == "landmark"].astype(str))
        adi_predictions = make_applicability_domain(data, database[key], list_genes_landmark, thresholds[key])
if True: # Prediction without z-score values but smiles
    # SMILES in bdd
    predictions = {}
    for key in database:
        if similarity_vega[key][similarity_vega[key]["VEGA_similarity"] == 1].shape[0] > 0:
            id_smiles = similarity_vega[key][similarity_vega[key]["VEGA_similarity"] == 1]["Unnamed: 0"].values[0]
            predictions[key] = make_predictions(database[key].loc[[id_smiles]], models, descriptors)
            adi_predictions = make_applicability_domain(database[key].loc[[id_smiles]], database[key], list_genes_landmark, thresholds[key])
        # smiles not in bdd using k-nn approach
        else:
            k = 6
            list_id_most_similars = similarity_vega[key].sort_values(by="VEGA_similarity", ascending=False)["Unnamed: 0"][:k]
            predictions[key] = make_predictions(database[key].loc[list_id_most_similars], models, descriptors)
            for key_pred in predictions[key]: # apply mean on predictions dictionary
                predictions[key][key_pred] = [sum(predictions[key][key_pred])/k]


# API Route
@app.route('/process', methods=['POST'])
def process_data():
    try:
        req_data = request.get_json()
        input_value = req_data.get("input", "")
        # Prediction with L1000 data
        result = [predictions, adi_predictions]
        # Prediction with k-NN approach
        #result = process_input_prediction(input_value)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
