from PIL import Image

def convert_rgb_to_grayscale(input_image_path, output_image_path):
    # 이미지 열기
    img = Image.open(input_image_path)
    # 그레이스케일로 변환
    grayscale_img = img.convert("L")
    # 변환된 이미지 저장
    grayscale_img.save(output_image_path)
    print(f"그레이스케일 이미지 저장 완료: {output_image_path}")

# 사용 예시
input_image = "Mel_VAD_TTS_record.jpg"
output_image = "Mel_VAD_TTS_record_gray.jpg"
convert_rgb_to_grayscale(input_image, output_image)
