from PIL import Image
import glob
import os

# 이미지 파일들의 경로
image_paths = glob.glob('CsponSpeech_Mel/*.png') 

output_dir = 'KsponSpeech_Mel/'

# 이미지 읽어오기
images = [Image.open(img_path) for img_path in image_paths]
cnt = 1
# 이미지를 순회하며 작업 수행
for img in images:
    img = img.convert("RGB")
    img.save(os.path.join(output_dir, f'c{cnt}.jpg'))
    cnt+=1
