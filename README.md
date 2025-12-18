회사에서 진행하는 팀원들 간 자체 역량강화 프로젝트,<br>
기존 잡다하게 작성한 코드들,<br>
공부 중인 코드들 등을 모아둔 repo 입니다.


### 환경변수 불러오기

```python 
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

### 가상환경 활성화 (PowerShell 기준)

```powershell

# 가상환경 만들기
py -3.13 -m venv env313

.\env313\Scripts\Activate

# 비활성화
deactivate
```

### Ubuntu 환경과 docker 연결

```bash
docker run -it `
  -v "C:\Users\LLOYDK:/workspace" `
  -w /workspace `
  -p 8000:8000 `
  ubuntu:24.04 bash

# 파이썬 설치
apt update && apt install -y git python3.12 python3.12-venv python3-pip vim

# 가상환경 세팅
python3.12 -m venv .venv_docker
source .venv_docker/bin/activate
```

### Github 기본 명령어 모음

```powershell
# 저장소 클론 (최초 1회)
git clone https://github.com/LLOYDK-BP-Tech/Demo.git
cd Demo

# 현재 브랜치 확인
git branch

# 브랜치 이동 (필요시)
git checkout main  # 혹은 다른 브랜치 이름

# 원격 브랜치 최신화
git fetch origin          # 원격 저장소 정보 동기화
git pull origin main       # 원격 main의 최신 커밋 반영

# 변경사항 스테이징 및 커밋
git add .                  # 모든 변경사항 추가
git commit -m "어쩌구"     # 커밋 메시지 작성

# 변경사항 푸시 (업로드)
git push -u origin main    # 원격 저장소(main)에 업로드

# add나 commit 한 거 지우고 싶을 때
git reset
```
