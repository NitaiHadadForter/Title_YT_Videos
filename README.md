# Title_YT_Videos

**App Features:**
- Single Video
  * YouTube video - given url, must be less than 10 minutes and 50Mb
  * Local Video - upload local video file
- Random Videos
  * Randomly choose 5 videos to predict out of chosen dataset - YouTube dataset (used for training), or TedTalk dataset
- For both modes the user can choose which fine-tuned model to use - small, base and flan
 
**Streamlint app link:** https://youtube-titles-generator.streamlit.app/

**Local App Run:**
- Clone the repository
- run `pip install -r requirements.txt`
- run `streamlit run app.py`

**Train:**
- Run finetune_models.ipynb notebook, preferably in high resources enviorment like Google Colab Pro L4
- Choose which model you want to finetune (only 1,2 and 3 are vailid inputs)
