

```markdown
# Skin Disease Classification App

This is a web application for classifying skin diseases using a pre-trained convolutional neural network model. The application allows users to upload images of skin conditions and provides predictions along with additional information about the disease.

## Features

- **Image Upload**: Users can upload images of skin conditions in JPG, JPEG, or PNG format.
- **Disease Prediction**: The model predicts the disease based on the uploaded image and provides a confidence score.
- **Information Display**: Users receive detailed information about the predicted disease, including symptoms, causes, and treatment options.
- **Chatbot Interaction**: Users can ask questions about their condition and receive answers from a chatbot powered by Google Generative AI.
- **User Feedback**: Users can provide feedback on their experience with the application.

## Technologies Used

- Python
- Streamlit
- Keras
- OpenCV
- NumPy
- Google Generative AI API
- gdown (for downloading model from Google Drive)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Navigate to the project directory:

   ```bash
   cd https://github.com/AayushKarwa/skin_analyzer
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have your Google API key ready and replace the placeholder in the code with your actual API key.

5. Ensure you have the model file's Google Drive link correctly set in the code.

## Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

This will start the web application, which you can access in your web browser at `http://localhost:8501`.

## Usage

1. Upload an image of a skin condition.
2. The application will display the predicted disease along with relevant information.
3. You can interact with the chatbot to ask questions related to your condition.
4. Provide feedback using the feedback section.

## Feedback

If you have any feedback, suggestions, or issues, please feel free to open an issue in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors and resources that made this project possible.
- The model used in this application was trained on a dataset of skin disease images.

```

