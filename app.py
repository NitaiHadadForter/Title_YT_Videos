import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import os
import subprocess
import json
from extract_audio import extract_youtube_audio, extract_audio
from extract_transcription import transcribe

# Constants
MAX_SIZE_MB = 50
MAX_DURATION_MINUTES = 10
TRANSCRIPTION_DIR = 'transcription_data'
DATASET_ID = "NitaiHadad/yt-titles-transcripts-clean"

# Define available models
MODELS = {
    "T5-Small Model": "NitaiHadad/video-to-titles-small",
    "T5-Base Model": "NitaiHadad/video-to-titles-base",
    "Flan-T5 Model": "NitaiHadad/video-to-titles-flan"
}


def check_video_constraints(url):
    """Check if video meets size and duration constraints"""
    try:
        # Use yt-dlp to get video info
        cmd = ['yt-dlp', '--dump-json', '--no-playlist', url]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Error getting video info: {result.stderr}")

        video_info = json.loads(result.stdout)

        # Get duration in minutes
        duration_minutes = video_info.get('duration', 0) / 60

        # Get filesize in MB
        size_mb = video_info.get('filesize_approx', 0) / (1024 * 1024)

        return {
            'title': video_info.get('title', 'Unknown'),
            'duration_minutes': duration_minutes,
            'size_mb': size_mb,
            'is_valid': duration_minutes <= MAX_DURATION_MINUTES and size_mb <= MAX_SIZE_MB,
            'error_message': (
                f"Video {'duration' if duration_minutes > MAX_DURATION_MINUTES else 'size'} "
                f"exceeds limit ({duration_minutes:.1f} min > {MAX_DURATION_MINUTES} min)"
                if duration_minutes > MAX_DURATION_MINUTES else
                f"({size_mb:.1f}MB > {MAX_SIZE_MB}MB)"
            ) if (duration_minutes > MAX_DURATION_MINUTES or size_mb > MAX_SIZE_MB) else None
        }
    except Exception as e:
        raise Exception(f"Error checking video: {str(e)}")


@st.cache_data
def load_huggingface_dataset():
    """Load the dataset from Hugging Face"""
    try:
        dataset = load_dataset(DATASET_ID, split='train')
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


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


def process_video(source, transcription_path):
    """Process video and return transcription path"""
    os.makedirs(transcription_path, exist_ok=True)
    audio_path = os.path.join(transcription_path, 'audio.wav')

    try:
        if isinstance(source, str) and source.startswith('https://www.youtube.com/'):
            # Check constraints before processing
            video_info = check_video_constraints(source)

            if not video_info['is_valid']:
                raise Exception(video_info['error_message'])

            video_id = source.split('watch?v=')[-1]
            audio_transcription_path = os.path.join(transcription_path, f'transcription-{video_id}.txt')

            # Process the video
            _, audio_path = extract_youtube_audio(source, transcription_path)
            st.info("Transcribing audio...")
            transcribe(audio_path, audio_transcription_path)

            # Clean up audio file
            if os.path.exists(audio_path):
                print(f"Removing audio file: {audio_path}")
                os.remove(audio_path)

            return audio_transcription_path

        else:
            file_name = source.name
            audio_transcription_path = os.path.join(transcription_path, f'transcription-{file_name}.txt')

            extract_audio(source, audio_path)
            st.info("Transcribing audio...")
            transcribe(audio_path, audio_transcription_path)

            if os.path.exists(audio_path):
                os.remove(audio_path)

            return audio_transcription_path

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise Exception(f"Error processing video: {str(e)}")


def main():
    st.title("YouTube Video Title Generator")

    # Initialize session state
    if 'current_model_id' not in st.session_state:
        st.session_state.current_model_id = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'tab' not in st.session_state:
        st.session_state.tab = "Single Video"

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
                     "T5-Small Model": "Lightweight model fine-tuned on T5-small, good for quick generations.",
                     "T5-Base Model": "Larger model with better generation quality but slower inference.",
                     "Flan-T5 Model": "Fine-tuned on Flan-T5 base model, optimized for natural language generation."
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

    # Main content - Tab Selection
    selected_tab = st.radio("", ["Single Video", "Random Videos"], horizontal=True, key="tab_selector")
    st.session_state.tab = selected_tab

    # Single Video Tab
    if st.session_state.tab == "Single Video":
        st.header("Generate title for a single video")
        input_type = st.radio("Select input type",
                              ["YouTube Link", "Local Video File"],
                              key="input_type_single")

        if input_type == "YouTube Link":
            video_url = st.text_input("Enter YouTube URL",
                                      "https://www.youtube.com/watch?v=dHy-qfkO54E",
                                      key="video_url_input")
            source = video_url

            # Add a check button
            if st.button("Check Video", key="check_video_button"):
                try:
                    with st.spinner("Checking video..."):
                        video_info = check_video_constraints(video_url)

                        # Show video information
                        st.info(
                            f"Video Information:\n"
                            f"Title: {video_info['title']}\n"
                            f"Duration: {video_info['duration_minutes']:.1f} minutes\n"
                            f"Estimated Size: {video_info['size_mb']:.1f}MB"
                        )

                        if video_info['is_valid']:
                            st.success("✅ Video meets all requirements")
                        else:
                            st.error(f"❌ {video_info['error_message']}")

                except Exception as e:
                    st.error(str(e))
                    st.warning("Please check if the URL is correct and the video is available.")
        else:
            video_file = st.file_uploader("Upload video file",
                                          type=['mp4', 'avi', 'mov'],
                                          key="video_file_uploader")
            source = video_file

            # Show file size for uploaded files
            if video_file:
                file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
                if file_size_mb > MAX_SIZE_MB:
                    st.error(f"File size ({file_size_mb:.1f}MB) exceeds limit of {MAX_SIZE_MB}MB")
                else:
                    st.success(f"File size ({file_size_mb:.1f}MB) is within limit")

        if st.button("Generate Title", key="generate_single_button"):
            if source:
                try:
                    with st.spinner("Processing video..."):
                        audio_transcription_path = process_video(source, TRANSCRIPTION_DIR)

                        input_text = 'summarize: ' + open(audio_transcription_path, "r").read()
                        titles = generate_titles(st.session_state.model,
                                                 st.session_state.tokenizer,
                                                 input_text,
                                                 token_max_length)
                        if os.path.exists(audio_transcription_path):
                            print(f"Removing transcription file: {audio_transcription_path}")
                            os.remove(audio_transcription_path)

                        st.subheader("Generated Titles:")
                        for i, title in enumerate(titles, 1):
                            st.write(f"{i}. {title}")

                except Exception as e:
                    st.error(str(e))
                    st.warning("Please try another video or check the URL.")

    # Random Videos Tab
    else:
        st.header("Generate titles for random videos")

        # Load dataset from Hugging Face
        dataset = load_huggingface_dataset()

        if dataset is not None:
            col1, col2 = st.columns(2)

            with col1:
                num_samples = st.number_input("Number of random samples",
                                              min_value=1,
                                              max_value=10,
                                              value=5,
                                              key="num_samples_random")

            with col2:
                if st.button("🔄 Generate Titles", key="generate_random_button"):
                    # Sample random examples
                    total_samples = len(dataset)
                    random_indices = torch.randint(0, total_samples, (num_samples,))
                    random_samples = [dataset[idx.item()] for idx in random_indices]

                    for i, sample in enumerate(random_samples):
                        st.write("---")
                        with st.expander(f"Sample: {sample['title'][:100]}...", expanded=True):
                            st.write("**Original title:**")
                            st.write(sample['title'])

                            with st.spinner("Generating titles..."):
                                input_text = 'summarize: ' + sample['transcript']
                                titles = generate_titles(st.session_state.model,
                                                         st.session_state.tokenizer,
                                                         input_text,
                                                         token_max_length)

                            st.write("**Generated titles:**")
                            for j, title in enumerate(titles, 1):
                                st.write(f"{j}. {title}")

                            # Add a "Show Transcript" option with a unique key
                            if st.checkbox("Show Transcript", key=f"transcript_{i}_{hash(sample['title'][:20])}"):
                                st.write("**Transcript:**")
                                st.write(sample['transcript'][:500] + "...")

            # Add dataset statistics
            with st.expander("Dataset Information"):
                st.write(f"Total number of samples: {len(dataset)}")
                st.write("Dataset source: Hugging Face")
                st.write("Dataset ID: " + DATASET_ID)
        else:
            st.error("Failed to load the dataset from Hugging Face. Please try again later.")
            st.info("You can still use the single video feature in the meantime.")


if __name__ == "__main__":
    st.set_page_config(page_title="YouTube Video Title Generator",
                       layout="wide")
    main()