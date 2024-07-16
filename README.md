# Mini-Projects
My experimental mini projects exploring Multimodal Gen AI, Quantum Computing lab projects that I develop on Qiskit and decentralized exchanges that I develop for my blockchain learning experience.

# Project 1 - Caption Generator

### Project Overview

The Caption Generator project is designed to generate descriptive captions for images using deep learning models. This involves several steps, including image preprocessing, feature extraction using convolutional neural networks (CNNs), and sequence prediction using recurrent neural networks (RNNs) or transformers.

### Code Explanation

#### 1. **Importing Libraries**

The notebook starts by importing necessary libraries, such as TensorFlow/Keras for building and training models, NumPy for numerical operations, and Matplotlib for visualizations.

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, add
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
```

#### 2. **Loading and Preprocessing Data**

The notebook includes code to load the image dataset and preprocess it. This involves resizing images, normalizing pixel values, and converting them into arrays suitable for model input.

```python
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x
```

#### 3. **Feature Extraction**

A pre-trained InceptionV3 model is used to extract features from images. The model is modified to remove the top layers, leaving the convolutional base for feature extraction.

```python
inception_model = InceptionV3(weights='imagenet')
model_new = Model(inception_model.input, inception_model.layers[-2].output)

def encode_image(img):
    img = preprocess_image(img)
    fea_vec = model_new.predict(img)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec
```

#### 4. **Captioning Model**

The project uses an RNN (LSTM in this case) to generate captions. The model consists of an embedding layer, an LSTM layer, and dense layers to predict the next word in the sequence.

```python
def build_model(vocab_size, max_length):
    inputs1 = tf.keras.Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)
    fe2 = RepeatVector(max_length)(fe1)
    
    inputs2 = tf.keras.Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256, return_sequences=True)(se1)
    
    decoder1 = add([fe2, se2])
    decoder2 = LSTM(256)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
```

#### 5. **Training the Model**

The notebook includes steps to train the model using the dataset. This involves feeding the image features and corresponding captions to the model and optimizing the weights.

```python
# Example of training code
model = build_model(vocab_size, max_length)
model.fit([features, sequences], targets, epochs=20, verbose=2)
```

#### 6. **Generating Captions**

After training, the model can generate captions for new images. The generation process involves using the model to predict the next word in the sequence until a stop token is generated or the maximum length is reached.

```python
def generate_caption(model, photo, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption
```

### Test Cases

Here are some test cases to demonstrate the functionality of the project:

#### Test Case 1: Generating a Caption for a New Image

```python
# Load and preprocess the image
image_path = 'path_to_image.jpg'
photo = encode_image(image_path)

# Generate caption
caption = generate_caption(model, photo, tokenizer, max_length)
print("Generated Caption:", caption)
```

#### Test Case 2: Evaluating Model Performance

```python
# Evaluate the model on a validation dataset
def evaluate_model(model, photos, descriptions, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        yhat = generate_caption(model, photos[key], tokenizer, max_length)
        actual.append([d.split() for d in desc_list])
        predicted.append(yhat.split())
    # Compute BLEU score
    bleu = corpus_bleu(actual, predicted)
    return bleu

bleu_score = evaluate_model(model, test_features, test_descriptions, tokenizer, max_length)
print("BLEU Score:", bleu_score)
```

The Caption Generator project combines image processing, feature extraction, and sequence prediction to generate descriptive captions for images. The notebook walks through loading data, preprocessing, building and training the model, and generating captions. By following the code and test cases, users can understand the workflow and customize the model for their specific needs.
