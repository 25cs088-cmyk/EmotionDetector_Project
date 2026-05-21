import watson_nlp

def emotion_detector(text_to_analyze):
    # Load the emotion detection model
    # Ensure this model is available in your Watson NLP environment
    emotion_model = watson_nlp.load('emotion_aggregated-workflow_lang_en_stock')
    
    # Run the model on the input text
    result = emotion_model.run(text_to_analyze)
    
    # Extract the emotion predictions
    # This returns a dictionary-like object containing scores for 
    # anger, disgust, fear, joy, and sadness
    emotions = result.get_emotion_predictions()
    
    return emotions
