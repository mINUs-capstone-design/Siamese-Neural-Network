from pydub import AudioSegment

def amplify_volume(input_file, output_file, amplification_factor):
    # 입력 파일 로드
    sound = AudioSegment.from_file(input_file)

    # 볼륨 증폭
    amplified_sound = sound + amplification_factor

    # 증폭된 볼륨을 가진 파일 저장
    amplified_sound.export(output_file, format="wav")

# 입력 파일과 출력 파일 지정
input_file = 'record.wav'
output_file = '10x_record.wav'

# 볼륨 증폭 계수 지정
amplification_factor = 20  # 원하는 증폭 정도로 조정

# 볼륨 증폭된 음성 데이터를 새로운 파일에 저장
amplify_volume(input_file, output_file, amplification_factor)