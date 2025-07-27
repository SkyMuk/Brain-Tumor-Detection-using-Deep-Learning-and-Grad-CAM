
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image and let the AI predict the tumor class.")

model = tf.keras.models.load_model("model.h5")

def preprocess_image(image):
    image = image.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(img_array, axis=0)

def make_gradcam(img_array, model, layer_name='block14_sepconv2_act'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    img = np.array(original_img.resize((299, 299)))
    heatmap = cv2.resize(heatmap, (299, 299))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    return Image.fromarray(np.uint8(superimposed))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI.", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### Prediction: `{predicted_class}` ({confidence:.2f}%)")

    heatmap = make_gradcam(img_array, model)
    result_img = overlay_heatmap(image, heatmap)

    st.image(result_img, caption="Grad-CAM Heatmap", use_column_width=True)
