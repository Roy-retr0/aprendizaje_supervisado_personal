from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo de forma segura verificando si el archivo existe
modelo_path = "modelo.pkl"
if os.path.exists(modelo_path):
    with open(modelo_path, 'rb') as file:
        modelo = pickle.load(file)
else:
    modelo = None

@app.route('/predecir', methods=['POST'])
def predict():
    # Validar si el modelo se cargó correctamente
    if modelo is None:
        return jsonify({'error': 'El modelo no está disponible o falta el archivo modelo.pkl'}), 500
    
    try:
        # Obtener JSON de la petición
        data = request.get_json(force=True)
        
        if not data or 'input' not in data:
            return jsonify({'error': 'Formato inválido. Se requiere una clave "input".'}), 400
        
        # Convertir los datos a un arreglo de Numpy con la forma correcta
        input_data = np.array(data['input']).reshape(1, -1)
        
        # Hacer predicción
        prediccion = modelo.predict(input_data)
        
        # Regresar predicción en formato JSON (convertido a entero nativo)
        return jsonify({'prediccion': int(prediccion[0])})
        
    except Exception as e:
        # Capturar cualquier error de procesamiento o dimensiones
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)