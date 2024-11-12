#!/bin/bash

cd /home/hanvit4303/gridtrader_backend


# 가상 환경 활성화 (경로는 실제 환경에 맞게 수정)
source /home/hanvit4303/gridtrader_backend/venv/bin/activate

# 최신 코드 가져오기
git pull

# 의존성 업데이트
#pip install -r requirements.txt

# 데이터베이스 마이그레이션
python manage.py migrate

# 정적 파일 수집
python manage.py collectstatic --noinput

# Gunicorn 재시작
sudo systemctl restart gunicorn

# Nginx 재시작 (필요한 경우)
# sudo systemctl restart nginx

echo "배포가 완료되었습니다."