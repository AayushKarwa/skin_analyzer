import json
import requests  # For making API calls
from keras.models import load_model 
from keras.optimizers import Adam
import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai

st.title("Skin Disease Classification")
API_KEY = 'AIzaSyD1dViBqAfFswyKFXFCW8RQ0w27tSHOcvk'
genai.configure(api_key=API_KEY)

# Load the model without compiling
model = load_model('final_vgg1920epochs.h5', compile=False)

# Compile the model with the desired optimizer and learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Open JSON file
with open('dat.json') as f:
    data = json.load(f)

keys = list(data)

# Streamlit session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def Predict(image):
    # Preprocess the image
    img = cv2.resize(image, (32, 32)) / 255.0
    prediction = model.predict(img.reshape(1, 32, 32, 3))
    
    # Get the predicted class and confidence score
    predicted_class_index = prediction.argmax()
    predicted_class = keys[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index] * 100  # Convert to percentage

    # Retrieve details from the JSON data
    description = data[predicted_class]['description']
    symptoms = data[predicted_class]['symptoms']
    causes = data[predicted_class]['causes']
    treatment = data[predicted_class]['treatement-1']

    return (
        predicted_class,
        description,
        symptoms,
        causes,
        treatment,
        f"{confidence_score:.2f}%"  # Format the confidence score
    )

# Disease Prediction section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))

    # Display the uploaded image
    st.image(image, channels="BGR", use_column_width=True)

    # Predict
    predicted_class, description, symptoms, causes, treatment, confidence_score = Predict(image)

    # Display results
    st.subheader("Results")
    st.write(f"**Name of Disease:** {predicted_class}")
    st.write(f"**Description:** {description}")
    st.write(f"**Symptoms:** {symptoms}")
    st.write(f"**Causes:** {causes}")
    st.write(f"**Treatment:** {treatment}")

    st.markdown('---')
    st.write('This Space predicts these diseases:\n \n1) Acne and Rosacea Photos. \n2) Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions.\n3) Eczema Photos. \n4) Melanoma Skin Cancer, Nevi, and Moles.\n5) Psoriasis pictures, Lichen Planus, and related diseases.\n6) Tinea, Ringworm, Candidiasis, and other Fungal Infections.\n7) Urticaria Hives.\n8) Nail Fungus and other Nail Diseases.')

    # Chatbot Input - displayed after results
    prompt = st.text_input('Ask about your disease here:')

    if prompt:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"You are a skin disease health chatbot named 'Skin Aura' who knows Acne and Rosacea Photos, "
            f"Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions, Eczema Photos, "
            f"Melanoma Skin Cancer Nevi and Moles, Psoriasis pictures Lichen Planus and related diseases, "
            f"Tinea Ringworm Candidiasis and other Fungal Infections, Urticaria Hives, Nail Fungus and other Nail Diseases. "
            f"You are a medical expert too. Please provide responses that are strictly related to health, medications, "
            f"and diseases. Ensure that all information is accurate, relevant, and presented in a clear and concise manner. "
            f"Tell the user that as the model predicted, you have {predicted_class} if prompt is asked. "
            f"The user has identified with {predicted_class} and has "
            f"{symptoms} symptoms. You can refer treatment from {treatment} link web scrape if you can from the link. "
            f"Now, based on the given prompt '{prompt}', respond with the information you have and also tell them the confidence score. "
            "Don't tell them you are an AI. React to greet messages with greetings and emoji and you are on version v1 testing."
        )

        # Store the chat history
        st.session_state.chat_history.append({"user": prompt, "bot": response.text})

    # Display chat history after prediction
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

    # "Know More" button section with feedback inside an expander
    with st.expander("Know More"):
        st.subheader("User Feedback")

        # Rating (1 to 5)
        rating = st.radio("How satisfied are you with this app?", [1, 2, 3, 4, 5], index=4)
        feedback = st.text_area("Share any additional feedback:", "")

        if st.button("Submit Feedback"):
            if feedback or rating:
                st.success("Thank you for your feedback!")

else:
    # Inform the user to upload an image
    st.warning("Please upload an image to get a prediction.")
