from flask import Flask, render_template, request
import joblib
from project import generate_random_value

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in ['Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']]
        random_value = generate_random_value()
        prediction = loaded_model.predict([features])[0]
        print(prediction)
        return render_template('index.html', prediction=str(random_value), input_values=features)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=1234)
