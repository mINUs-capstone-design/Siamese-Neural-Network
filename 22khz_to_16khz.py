from pydub import AudioSegment

def convert_sample_rate(input_file_path, output_file_path, target_sample_rate=16000):
    # 오디오 파일 로드
    audio = AudioSegment.from_wav(input_file_path)
    
    # 샘플 레이트 변환
    audio = audio.set_frame_rate(target_sample_rate)
    
    # 변환된 오디오 파일 저장
    audio.export(output_file_path, format='wav')

# 사용 예시
input_file_path = "api_test/VAD_00383-F-98-ZH-A-ATQ005-0075387.wav"
output_file_path = 'api_test/16khz_2.wav'  # 변환된 파일을 저장할 경로
convert_sample_rate(input_file_path, output_file_path)

print(f"Converted wav file saved to {output_file_path}")
