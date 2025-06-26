from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model pipeline (includes vectorizer + classifier)
model = joblib.load('spam/spam_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict([message])[0]
        result = "SPAM ❌" if prediction == 1 else "NOT SPAM ✅"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
