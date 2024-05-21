from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file_path, wav_file_path):

    audio = AudioSegment.from_mp3(mp3_file_path)

    audio.export(wav_file_path, format="wav")
    print(f"파일 변환 완료 : {wav_file_path}")


mp3_file = "Record_man.mp3"
wav_file = "record_man.wav"
convert_mp3_to_wav(mp3_file, wav_file)
