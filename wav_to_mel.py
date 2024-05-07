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

# VAD 알고리즘
def wav_to_vad(wav_data, path):
    y, sr = librosa.load(wav_data)
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    sf.write(path, y_trimmed, sr)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 현재 위치한 디렉토리 내의 확장자가 .pcm인 모든 파일 불러오기
data_dir = "/content/drive/MyDrive/asia"
save_dir = "/content/drive/MyDrive/test_asia_VAD"

original_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

# 반복하여 한번에 여러 파일 처리
for original_file in original_files:
    # original 파일의 이름
    original_path = os.path.join(data_dir, original_file)

    # VAD 적용하여 새로운 .wav 파일 저장
    vad_wav_path = os.path.join(save_dir, f"VAD_{os.path.splitext(original_file)[0]}.wav")
    wav_to_vad(original_path, vad_wav_path)
    original_path = vad_wav_path
    print(".wav 파일에 VAD 알고리즘 적용 완료")

    # WAV 파일 읽기
    y, sr = librosa.load(original_path)

    # Mel-spectrogram 계산
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Mel-spectrogram을 데시벨로 변환
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Mel-spectrogram 플로팅
    plt.figure(figsize=(10, 4)) # img 사이즈 변경
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
    output_filename = os.path.join(save_dir, f"Mel_spectrum_{os.path.splitext(os.path.basename(original_path))[0]}.jpg")
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(".wav 파일에 Mel 알고리즘 적용 완료")
