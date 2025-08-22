import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon=" ",
    layout="centered"
)


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("saved_models/1.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    # image_array = image_array / 255.0 
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_disease(model, image_array):
    predictions = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence


def display_results(predicted_class, confidence):
    st.markdown("#### 🔎 Prediction Results", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="padding:10px 0;">
            <span style="font-size:18px;"><b>Predicted Disease:</b></span><br>
            <span style="font-size:28px; color:#f63366;"><b>{predicted_class}</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="padding:10px 0;">
            <span style="font-size:18px;"><b>Confidence:</b></span><br>
            <span style="font-size:22px; color:#25C2A0;"><b>{confidence:.2f}%</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if predicted_class == "Healthy":
        st.success("✅ This potato leaf appears healthy!")
        st.balloons()
    else:
        st.markdown(
            f"""
            <div style="background-color:#fff3cd; border-left: 8px solid #ffe066; padding:10px 15px; margin-bottom:8px;">
            <span style="color:#856404; font-weight:600;">⚠️ This leaf shows signs of <b>{predicted_class}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if predicted_class == "Early Blight":
            st.info("💡 **Tip:** Remove affected leaves, improve air circulation, and consider fungicide treatment.", icon="💡")
        elif predicted_class == "Late Blight":
            st.info("💡 **Tip:** Serious disease. Remove and destroy affected plants immediately to prevent spread.", icon="💡")


def main():
    st.title("📁 Potato Leaf Disease Classifier")
    st.markdown("Upload an image of a potato leaf to detect diseases.")

    model = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.image(image, caption="Uploaded Image", width=350, use_container_width=False)
        with col2:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            image_array = preprocess_image(image)
            predicted_class, confidence = predict_disease(model, image_array)
            display_results(predicted_class, confidence)


if __name__ == "__main__":
    main()
