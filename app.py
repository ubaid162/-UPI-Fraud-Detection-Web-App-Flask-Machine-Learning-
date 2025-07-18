from flask import Flask, render_template, request
import pickle
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('model/rfc_model.pkl', 'rb'))
encoders = pickle.load(open('model/encoders.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Fake user database
users = {
    "admin": "admin123",
    "ubaid": "1234",
    "guest": "guest"
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            return render_template('index.html')  # Show form if login success
        else:
            error = "❌ Incorrect username or password. Please try again."
            return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect cleaned user inputs
    data = {
        'Amount': float(request.form['Amount']),
        'MerchantCategory': request.form['MerchantCategory'],
        'TransactionType': request.form['TransactionType'],
        'Latitude': float(request.form['Latitude']),
        'Longitude': float(request.form['Longitude']),
        'AvgTransactionAmount': float(request.form['AvgTransactionAmount']),
        'UnusualLocation': request.form.get('UnusualLocation') == 'yes',
        'UnusualAmount': request.form.get('UnusualAmount') == 'yes',
        'NewDevice': request.form.get('NewDevice') == 'yes',
        'FailedAttempts': int(request.form['FailedAttempts']),
        'BankName': request.form['BankName'],
        'Transaction_frequency': request.form['Transaction_frequency']
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

   
    for col in  ['MerchantCategory', 'TransactionType','UnusualLocation','UnusualAmount','NewDevice', 'BankName']:
        df[col] = encoders[col].transform(df[[col]])



    df['Transaction_frequency'] = encoders['Transaction_frequency'].transform(df[['Transaction_frequency']].values)
    # Predict
    # Step 5: DEBUG PRINT
    print("========= PREDICTION DEBUG INFO =========")
    print(df)
    print("Model input shape:", df.shape)
    print("Prediction (raw):", model.predict(df))
    print("Probabilities:", model.predict_proba(df))
    print("=========================================")


    expected_order = ['Amount', 'MerchantCategory', 'TransactionType', 'Latitude',
                  'Longitude', 'AvgTransactionAmount', 'UnusualLocation', 'UnusualAmount',
                  'NewDevice', 'FailedAttempts', 'BankName', 'Transaction_frequency']
    df = df[expected_order]

    # Step 6: Predict
    prediction = int(model.predict(df)[0])
    print("Prediction is: ",prediction)
     
    if prediction == 1:
        prediction_status = "fraud"
    else:
        prediction_status = "safe"


    return render_template('result.html', status=prediction_status)



if __name__ == '__main__':
    app.run(debug=True)

