"""
<각 파일 이름>
KsponSpeech_[숫자].pcm : .wav로 변환 전 원본 .pcm 파일
KsponSpeech_[숫자].wav : .pcm -> .wav로 변환된 파일
VAD_KsponSpeech_[숫자].wav : VAD 알고리즘 적용된 .wav 파일
Mel_spectrum_KsponSpeech_[숫자].jpg : RGB 3채널 Mel 적용 이미지
"""

# ----------------------------------------------------------------
import os
import numpy as np
import librosa
import librosa.effects
import librosa.display
import soundfile as sf
import wave
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# .pcm -> .wav 변환함수
def pcm_to_wav(pcm_file, wav_file, channels, sample_width, frame_rate):
    # PCM 파일 열기
    with open(pcm_file, 'rb') as pcm:
        # WAV 파일 열기
        with wave.open(wav_file, 'wb') as wav:
            # WAV 파일 헤더 설정
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(frame_rate)
            
            # 오디오 데이터 쓰기
            data = pcm.read()
            wav.writeframes(data)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# VAD 알고리즘
def wav_to_vad(no_vad_wav, apply_vad_wav):
    y, sr = librosa.load(no_vad_wav)
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    sf.write(apply_vad_wav, y_trimmed, sr)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 현재 위치한 디렉토리 내의 확장자가 .pcm인 모든 파일 불러오기
pcm_directory = "/content/drive/MyDrive/kr"

my_pcm_data = [f for f in os.listdir(pcm_directory) if f.endswith(".pcm")]

# 반복하여 한번에 여러 파일 처리
for pcm_file in my_pcm_data:
    # .pcm 파일의 경로
    pcm_sound = os.path.join(pcm_directory, pcm_file)
    
    # .wav 파일의 경로
    wav_file = os.path.join(pcm_directory, os.path.splitext(pcm_file)[0] + ".wav")
    
    # PCM을 WAV로 변환
    pcm_to_wav(pcm_sound, wav_file, channels=1, sample_width=2, frame_rate=16000)
    print(".pcm을 .wav로 변환 완료")
    
    # VAD 적용하여 새로운 .wav 파일 저장
    vad_wav_file = os.path.join(pcm_directory, f"VAD_{os.path.splitext(pcm_file)[0]}.wav")
    wav_to_vad(wav_file, vad_wav_file)
    print(".wav 파일에 VAD 알고리즘 적용 완료")
    
    # WAV 파일 읽기
    y, sr = librosa.load(wav_file)
    
    # Mel-spectrogram 계산
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Mel-spectrogram을 데시벨로 변환
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Mel-spectrogram 플로팅
    plt.figure(figsize=(10, 4))
    axes = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    
    # plot 그래프 이미지만 나타내기
    plt.yticks(ticks= []) # y축 tick 제거
    plt.xticks(ticks= []) # x축 tick 제거
    
    # # x축, y축 각각 눈금 & 테두리 제거
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 제목과 컬러바(오른쪽 색상표시바) 제거
    plt.title('')
    plt.colorbar().remove()

    # 이미지 저장 (그래프 내용 부분만...흰 배경X)
    output_filename = os.path.join(pcm_directory, f"Mel_spectrum_{os.path.splitext(os.path.basename(wav_file))[0]}.jpg")
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(".wav 파일에 Mel 알고리즘 적용 완료")
