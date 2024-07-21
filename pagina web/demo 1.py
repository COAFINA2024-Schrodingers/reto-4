from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    resultado = "Este es el resultado de tu código Python"
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
