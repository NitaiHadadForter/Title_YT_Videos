from multiprocessing import Process
from faster_whisper import WhisperModel

def transcribe_impl(audio_path, output_path):
    model = WhisperModel("small", compute_type="int8")
    print("Starting transcription")
    segments, info = model.transcribe(audio_path)
    print("Transcription language", info[0])
    segments = list(segments)
    with open(output_path, "w") as outfile:
        print("Writing transcription to", output_path)
        outfile.write("\n".join([s.text for s in segments]))
    print("Finished transcription")

def transcribe(audio_path, output_path):
    p = Process(target=transcribe_impl, args=[audio_path, output_path])
    p.start()
    p.join()
    p.close()
