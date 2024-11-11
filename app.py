import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import os
from extract_audio import extract_youtube_audio, extract_audio
from extract_transcription import transcribe

# Define available models
MODELS = {
    "Flan-T5 Model": "NitaiHadad/video-to-titles-flan",
    "T5-Small Model": "NitaiHadad/video-to-titles-small",  # Replace with your actual repo
    "T5-Base Model": "NitaiHadad/video-to-titles-base"  # Replace with your actual repo
}


@st.cache_resource
def load_model(model_id, use_custom_model=False, custom_model_path=None):
    """Load the model and tokenizer"""
    try:
        if use_custom_model and custom_model_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(custom_model_path)
            tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def generate_titles(model, tokenizer, input_text, token_max_length=512):
    """Generate titles using the loaded model"""
    with torch.no_grad():
        tokenized_text = tokenizer(input_text, truncation=True, padding=True,
                                   return_tensors='pt', max_length=token_max_length)
        source_ids = tokenized_text['input_ids']
        source_mask = tokenized_text['attention_mask']

        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_new_tokens=20,
            num_beams=5,
            repetition_penalty=1,
            length_penalty=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
            num_return_sequences=5
        )

        return [tokenizer.decode(g, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for g in generated_ids]


def main():
    st.set_page_config(page_title="YouTube Video Title Generator",
                       layout="wide")

    st.title("YouTube Video Title Generator")

    # Initialize session state
    if 'current_model_id' not in st.session_state:
        st.session_state.current_model_id = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None

    # Sidebar for model selection
    st.sidebar.header("Model Settings")

    # Model selection
    selected_model = st.sidebar.radio(
        "Select Model",
        list(MODELS.keys()),
        help="Choose between different pre-trained models"
    )

    # Custom model option
    use_custom_model = st.sidebar.checkbox("Use custom model")
    if use_custom_model:
        custom_model_path = st.sidebar.text_input("Enter model path")
        if custom_model_path and os.path.exists(custom_model_path):
            model_path = custom_model_path
        else:
            st.sidebar.error("Invalid model path")
            return
    else:
        model_path = MODELS[selected_model]

    # Add model information
    with st.sidebar.expander("Model Information"):
        st.write({
                     "Flan-T5 Model": "Fine-tuned on Flan-T5 base model, optimized for natural language generation.",
                     "T5-Small Model": "Lightweight model fine-tuned on T5-small, good for quick generations.",
                     "T5-Base Model": "Larger model with better generation quality but slower inference."
                 }[selected_model])

    # Token length setting
    token_max_length = st.sidebar.number_input("Token max length",
                                               min_value=128,
                                               max_value=1024,
                                               value=512)

    # Load model if it's different from the current one
    if st.session_state.current_model_id != model_path:
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.model, st.session_state.tokenizer = load_model(
                model_path,
                use_custom_model,
                custom_model_path if use_custom_model else None
            )
            st.session_state.current_model_id = model_path
        st.success(f"{selected_model} loaded successfully!")

    # Main content
    tab1, tab2 = st.tabs(["Single Video", "Random Videos"])

    with tab1:
        st.header("Generate title for a single video")
        input_type = st.radio("Select input type",
                              ["YouTube Link", "Local Video File"])

        if input_type == "YouTube Link":
            video_url = st.text_input("Enter YouTube URL",
                                      "https://www.youtube.com/watch?v=dHy-qfkO54E")
            source = video_url
        else:
            video_file = st.file_uploader("Upload video file",
                                          type=['mp4', 'avi', 'mov'])
            source = video_file

        if st.button("Generate Title"):
            if source:
                with st.spinner("Processing video..."):
                    # Process audio and transcription
                    audio_path = 'transcription data/audio.wav'
                    transcription_path = 'transcription data/'

                    # Create directory if it doesn't exist
                    os.makedirs('transcription data', exist_ok=True)

                    if isinstance(source, str) and source.startswith('https://www.youtube.com/'):
                        video_id = source.split('watch?v=')[-1]
                        audio_transcription_path = transcription_path + f'transcription-{video_id}.txt'
                        if not os.path.exists(audio_transcription_path):
                            _, audio_path = extract_youtube_audio(source, transcription_path)
                            st.info("Transcribing audio...")
                            transcribe(audio_path, audio_transcription_path)
                    else:
                        file_name = source.name
                        audio_transcription_path = transcription_path + f'transcription-{file_name}.txt'
                        if not os.path.exists(audio_transcription_path):
                            extract_audio(source, audio_path)
                            st.info("Transcribing audio...")
                            transcribe(audio_path, audio_transcription_path)

                    # Generate titles
                    input_text = 'summarize: ' + open(audio_transcription_path, "r").read()
                    titles = generate_titles(st.session_state.model,
                                             st.session_state.tokenizer,
                                             input_text,
                                             token_max_length)

                    # Display results
                    st.subheader("Generated Titles:")
                    for i, title in enumerate(titles, 1):
                        st.write(f"{i}. {title}")

    with tab2:
        st.header("Generate titles for random videos")
        if os.path.exists("YT-titles-transcripts-clean.csv"):
            if 'dataset' not in st.session_state:
                st.session_state.dataset = pd.read_csv("YT-titles-transcripts-clean.csv")
            if st.button("Generate Titles for 5 Random Videos"):
                random_videos = st.session_state.dataset.sample(n=5)

                for _, row in random_videos.iterrows():
                    st.write("---")
                    st.write(f"**Original title:** {row['title']}")

                    titles = generate_titles(st.session_state.model,
                                             st.session_state.tokenizer,
                                             row['transcript'],
                                             token_max_length)

                    st.write("**Generated titles:**")
                    for i, title in enumerate(titles, 1):
                        st.write(f"{i}. {title}")
        else:
            st.warning("Random video dataset not found!")


if __name__ == "__main__":
    main()