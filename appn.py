from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import fitz  # PyMuPDF
import re

app = Flask(__name__)

# Load your custom TensorFlow model for image predictions
model = tf.keras.models.load_model('my_model_resnet152.h5')

# Initialize the GPT-2 model and tokenizer for text processing
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model_text = GPT2LMHeadModel.from_pretrained(model_name)

disease_names = {
   0: 'Light Diseases and Disorders of Pigmentation d',
    1:'Lupus and other Connective Tissue diseases d',
    2:'Acne and Rosacea d',
    3:'Systemic Disease',
    4:'Poison Ivy and other Contact Dermatitis d',
    5:'Vascular Tumors',
    6:'Urticaria Hives d',
    7:'Atopic Dermatitis',
    8:'Bullous Disease d',
    9:'Hair Loss Alopecia and other Hair Diseases',
    10:'Tinea Ringworm Candidiasis and other Fungal Infections',
    11:'Psoriasis Lichen Planus and related diseases',
    12:'Melanoma Skin Cancer Nevi and Moles d',
    13:'Nail Fungus and other Nail Disease',
    14:'Scabies Lyme Disease and other Infestations and Bites',
    15:'Eczema d'
}

def preprocess_image(img_path):
    # Your previously provided preprocessing function
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img)
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = img.astype('float32') / 225.0
    return img

# def predict_image(image_path, model):
#     # Your prediction function as provided
#         img = preprocess_image(image_path)
#         img = np.expand_dims(img, axis=0)
#         predictions = model.predict(img)
#         predictions = predictions.flatten()
#         threshold = 0
#         high_conf_predictions = [(i, p) for i, p in enumerate(predictions) if p >= threshold]
#         high_conf_predictions.sort(key=lambda x: x[1], reverse=True)
#
#         # Map class indices to disease names
#         high_conf_predictions = [(disease_names[class_index], p) for class_index, p in high_conf_predictions]
#         return high_conf_predictions
def predict_image(image_path, model):
    img = preprocess_image(image_path)

    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predictions = predictions.flatten()
    high_conf_predictions = [(i, p) for i, p in enumerate(predictions) if p >= 0]  # Assuming you want to consider all predictions
    high_conf_predictions.sort(key=lambda x: x[1], reverse=True)

    # Limit to top 3 predictions and map class indices to disease names
    top_predictions = high_conf_predictions[:1]  # Get top 3 predictions
    top_disease_names = [disease_names[class_index] for class_index, _ in top_predictions]
    return top_disease_names
#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         if 'imagefile' in request.files:
#             imagefile = request.files['imagefile']
#             image_path = "./static/Image/temp/" + imagefile.filename
#             if not os.path.exists('./static/Image/temp/'):
#                 os.makedirs('./static/Image/temp/')
#             imagefile.save(image_path)
#
#             predictions = predict_image(image_path, model)
#             prediction_text =  "\n".join([f"Class {class_index}" for class_index in predictions])
#
#             return render_template('index.html', prediction=prediction_text, image='../static/Image/temp/' + imagefile.filename)
#     # If not POST or no imagefile in request, render the upload form
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'imagefile' in request.files:
            imagefile = request.files['imagefile']
            image_path = os.path.join('static/Image/temp', imagefile.filename)
            print(imagefile.filename)
            text = ""



            if not os.path.exists('static/Image/temp'):
                os.makedirs('static/Image/temp')
            imagefile.save(image_path)

            top_disease_names = predict_image(image_path, model)
            prediction_text = "\n".join(top_disease_names)  # Join the top disease names
            # print(top_disease_names)
            if(len(text) == 0):
                text = prediction_text
            return render_template('index.html', prediction=text, image='../static/Image/temp/' + imagefile.filename)
    # If not POST or no imagefile in request, render the upload form
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    if question:
        # Context for GPT-2 model can be loaded or defined here
        context = "Some context if needed or load from somewhere"
        answer = generate_answer(question, context)
        te = answer.split('Answer:')[-1]
        return jsonify({"answer": te})
    return jsonify({"error": "Please enter a question."}), 400

# def generate_answer(question, context):
#     input_text = f"Context: {context[:1024]}\nQuestion: {question}\nAnswer:"
#     input_ids = tokenizer.encode(input_text, return_tensors='pt')
#     output = model_text.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
#     answer = tokenizer.decode(output[0], skip_special_tokens=True)
#     return answer
def generate_answer(question, context):
    input_text = f"Context: {context[:1024]}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # You may adjust max_length further if needed
    output = model_text.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2,pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text
    answer_full = tokenizer.decode(output[0], skip_special_tokens=True)

    # Process to stop after three full stops
    full_stops = answer_full.count('.')
    if full_stops >= 3:
        # Find the position of the third full stop
        stop_index = [pos for pos, char in enumerate(answer_full) if char == '.'][2]
        # Slice the answer to include text up to the third full stop
        answer = answer_full[:stop_index + 1]
    else:
        # If there are fewer than 3 full stops, use the full answer
        answer = answer_full

    return answer


if __name__ == '__main__':
    app.run(port=5500, debug=True)