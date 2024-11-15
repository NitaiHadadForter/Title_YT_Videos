import os
import ffmpeg
import yt_dlp
import tempfile



def extract_audio(uploaded_file, output_path):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        # Extract audio using the saved temporary file path
        stream = ffmpeg.input(temp_file_path)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream, overwrite_output=True)
        return output_path
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def extract_youtube_audio(link, data_folder):
    print("downloading audio...")
    audio_path = os.path.join(data_folder, 'audio.wav')
    try:
        os.remove(audio_path)
    except OSError as e:
        print("%s - %s - Need to download audio." % (e.filename, e.strerror))
    ydl_opts = {
        'extract_audio': True,
        'format': 'bestaudio',
        'outtmpl': audio_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=True)
        video_title = info_dict['title']
        ydl.download(link)
    print("finished downloading")

    print("audio saved to: ", audio_path)

    return video_title, audio_path
