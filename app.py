from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# --- FASE CTRL PRO: TREINANDO O ROBÔ ---
# 1. Lemos os dados históricos do Titanic
df = pd.read_csv('titanic.csv')

# 2. Transformamos texto em números (Robôs preferem números)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 3. Separamos as características (X) do resultado (y)
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# 4. Criamos e treinamos o "Cérebro"
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X, y)

# --- FASE INTEGRAÇÃO: O SERVIDOR ---
@app.route('/')
def home():
    # Mostra o arquivo HTML (O Rosto)
    return render_template('index.html')

@app.route('/prever', methods=['POST'])
def prever():
    # Pega os dados que vieram do formulário HTML
    idade = float(request.form['idade'])
    sexo = 1 if request.form['sexo'] == 'female' else 0
    classe = int(request.form['classe'])
    
    # Pede para a IA prever o futuro do usuário
    previsao = modelo.predict([[classe, sexo, idade]])[0]
    
    # Define a resposta visual
    if previsao == 1:
        mensagem = "🎉 VOCÊ SOBREVIVEU! 🎉<br>Um bote salva-vidas estava te esperando."
        cor = "#4caf50"
    else:
        mensagem = "🧊 GLUB GLUB... 🧊<br>A água estava muito gelada."
        cor = "#f44336"
        
    return f"""
    <body style="background-color: {cor}; color: white; font-family: Arial; text-align: center; padding-top: 100px;">
        <h1>{mensagem}</h1>
        <br><br>
        <a href="/" style="color: white; text-decoration: underline; font-size: 20px;">Tentar outra vida</a>
    </body>
    """

if __name__ == '__main__':
    app.run(debug=True)