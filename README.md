# Title_YT_Videos

**App Features:**
- Single Video
  * YouTube video - given url, must be less than 10 minutes and 50MB
  * Local Video - upload
- Random Videos
  * Randomly choose 5 videos to predict, out of 1,004 videos
- Can choose which fine-tuned model to use - small, base and flan
 
**Streamlint app link:** https://youtube-titles-generator.streamlit.app/

* App instruction *
The application has two modes of operation:
1. Single video - youtube link or local video file. The model works on video in size of less than 50Mb and length less than 10 minutes, to avoid memory issues.
2. Random video - chose 5 youtube videos randomly, out of 1,000 videos we stored. In this mode, we skip the transcribing phase and use existing transcription.
For both modes, the user can choose which model to use (Base is the most accurate, small is the fastest), and set the length of the transcript that will be used in the model.


**Local App Run:**
- Clone the repository
- run `pip install -r requirements.txt`
- run `streamlit run app.py`

**Train:**
- Run t5_fintune_model notebook
