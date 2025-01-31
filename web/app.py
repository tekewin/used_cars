from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained regression model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = joblib.load(file)

# Predefined brand and model lists
brands = ['Ford', 'BMW', 'Mercedes-Benz', 'Chevrolet', 'Porsche', 'Audi', 'Toyota', 'Lexus', 'Jeep', 'Land']
models = {
    'Ford': ['F-150 XLT', 'Mustang GT Premium', 'F-250 Lariat', 'Explorer XLT', 'F-150 Lariat'],
    'BMW': ['M3 Base', 'M4 Base', 'M5 Base', 'X7 xDrive40i', 'X5 xDrive35i'],
    'Mercedes-Benz': ['E-Class E 350 4MATIC', 'Metris Base', 'SL-Class SL 550', 'G-Class G 550 4MATIC'],
    'Chevrolet': ['Corvette Base', 'Camaro 2SS', 'Corvette Stingray w/2LT', 'Suburban LT', 'Suburban Premier'],
    'Porsche': ['911 Carrera', '911 Carrera S', 'Macan S', '911 Carrera 4S', '911 GT3'],
    'Audi': ['Q5 2.0T Premium Plus', 'Q5 S line Premium Plus', 'S4 3.0T Premium Plus', 'A4 2.0T Premium', 'A3 2.0T Premium'],
    'Toyota': ['Land Cruiser Base', 'Highlander XLE', 'Sequoia Limited', 'FJ Cruiser Base', 'Tundra SR5'],
    'Lexus': ['ES 350 Base', 'GX 460 Base', 'RX 350 Base', 'LC 500 Base', 'LX 570 Three-Row'],
    'Jeep': ['Wrangler Sport', 'Wrangler Unlimited Sport', 'Wrangler Unlimited Sahara', 'Gladiator Overland', 'Grand Cherokee Limited'],
    'Land': ['Rover Range Rover Sport HSE', 'Rover Range Rover Sport Supercharged', 'Rover Range Rover 3.0L V6 Supercharged HSE', 'Rover Range Rover 3.0L Supercharged HSE', 'Rover Range Rover 5.0L Supercharged']
}

@app.route('/')
def index():
    return render_template("index.html", brands=brands, models=models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['engine_hp'] = pd.to_numeric(df['engine_hp'], errors='coerce')

    print(df)

    predicted_log_price = model.predict(df)[0]

    # convert from log_prediction
    prediction = np.expm1(predicted_log_price)

    formatted_price = f"${prediction:,.2f}"
    return jsonify({"predicted_price": formatted_price})

if __name__ == '__main__':
    app.run(debug=True)
