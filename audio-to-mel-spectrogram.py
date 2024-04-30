# ----------------------------------------------------------------
#1
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import wave
from PIL import Image
import glob
# ----------------------------------------------------------------

# ----------------------------------------------------------------
#2
# .pcm -> .wav 변환하는 파트
def pcm_to_wav(pcm_file, wav_file, channels, sample_width, frame_rate):
    # PCM 파일 열기
    with open(pcm_file, 'rb') as pcm:
        # PCM 파일의 오디오 데이터 길이 확인 -> 이 부분이 없으면 배속으로 저장됨
        data = pcm.read()
        pcm_length = len(data)
        
        # 오디오의 샘플 수 계산
        num_samples = pcm_length // (channels * sample_width)
        
        # WAV 파일 열기
        with wave.open(wav_file, 'wb') as wav:
            # WAV 파일 헤더 설정
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(frame_rate)
            wav.setnframes(num_samples)
            
            # 오디오 데이터 쓰기
            wav.writeframes(data)
# ----------------------------------------------------------------

# 현재 실행 중인 파일의 경로
current_directory = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------
# 이미지 변환 함수
def change_picture():
    input_dir = '.'
    output_dir = '.'
    
    # 이미지 읽어오기
    images = [file for file in os.listdir(current_directory) if file.endswith(".png")]
    cnt = 1
    
    # 이미지를 순회하며 작업 수행
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path)  # 이미지 열기
        img = img.convert("RGB")  # 이미지를 RGB 형식으로 변환
        img.save(os.path.join(output_dir, f'c_{cnt}.jpg'))  # JPG로 저장
        cnt += 1
# ----------------------------------------------------------------

# ----------------------------------------------------------------
#3
# 현재 위치한 디렉토리 내의 확장자가 .pcm인 모든 파일 불러오기
pcm_directory = "."

my_pcm_data = [f for f in os.listdir('.') if f.endswith(".pcm")]

# 반복하여 한번에 여러 파일 처리
for pcm_file in my_pcm_data:
    # .pcm 파일의 경로
    pcm_sound = os.path.join(pcm_directory, pcm_file)
    
    # .wav 파일의 경로
    wav_file = os.path.join(pcm_directory, os.path.splitext(pcm_file)[0] + ".wav")
    pcm_to_wav_sound = os.path.join(pcm_directory, wav_file)
    
    # PCM을 WAV로 변환
    pcm_to_wav(pcm_sound, pcm_to_wav_sound, channels=1, sample_width=2, frame_rate=16000)
    
    # WAV 파일 읽기
    y, sr = librosa.load(pcm_to_wav_sound)
    
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
    output_filename = os.path.join(pcm_directory, f"Mel_spectrum_{os.path.splitext(os.path.basename(pcm_to_wav_sound))[0]}.png")
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 이미지 변환 함수 호출
    change_picture()
    print("이미지 변환 완료")
# ----------------------------------------------------------------
