from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt

# 녹음된 파일 불러오기
audio = AudioSegment.from_wav("record.wav")


noise_sample_duration = 1000
noise_sample = audio[:noise_sample_duration]

# AudioSegment를 numpy 배열로 변환하는 함수
def audiosegment_to_numpy(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples

# 노이즈 샘플과 전체 오디오를 numpy 배열로 변환
noise_data = audiosegment_to_numpy(noise_sample)
data = audiosegment_to_numpy(audio)

# 오디오의 샘플레이트 가져오기
rate = audio.frame_rate

# 노이즈 감소 적용
reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=noise_data)

# numpy 배열을 다시 AudioSegment로 변환하는 함수
def numpy_to_audiosegment(data, frame_rate, sample_width, channels):
    return AudioSegment(
        data.tobytes(), 
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )

# 노이즈 감소된 numpy 배열을 다시 AudioSegment로 변환
reduced_noise_audio = numpy_to_audiosegment(
    reduced_noise, 
    frame_rate=rate,
    sample_width=data.dtype.itemsize, 
    channels=audio.channels
)

# 필터링된 오디오 저장
reduced_noise_audio.export("filtered_record.wav", format="wav")

# 필터링된 오디오 재생 (필요 시)
# play(reduced_noise_audio)

# 필터링된 오디오를 시각화하여 확인
plt.plot(reduced_noise)
plt.title("filtered_record.wav")
plt.show()