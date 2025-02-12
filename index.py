import pickle
from flask import Flask, request, jsonify

# Load the saved model
with open('gb_model_without_random_state.pkl', 'rb') as model_file:
    loaded_model1 = pickle.load(model_file)

# Create a Flask web application
app = Flask("performancePrediction_gbApi")

# Endpoint to make predictions using query parameters
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get feature1 and feature2 from query parameters in the URL
        D1 = float(request.args.get('d1'))
        D2 = float(request.args.get('d2'))
        D3 = float(request.args.get('d3'))
        D4 = float(request.args.get('d4'))
        D5 = float(request.args.get('d5'))
        D6 = float(request.args.get('d6'))
        D7 = float(request.args.get('d7'))

        # Make predictions using the loaded model and the provided features
        predictions = loaded_model.predict([[D1, D2, D3, D4, D5, D6, D7]])

        # Prepare the response
        response = {'predictions': predictions.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


# if __name__ == "__main__":
#     app.run()(debug=False,host='0.0.0.0')
