# PhotoRefSorter v0.7 — Folder First / Portable

대량의 사진·영상 폴더를 지정하면 얼굴을 찾아 등록된 인물과 비교하고, 인물별 결과 폴더로 자동 분류하는 Windows용 로컬 사진 관리 앱입니다.

## v0.7 핵심 변화

### 1. 폴더 분석이 앱의 첫 화면
- 첫 화면에서 바로 `분석할 폴더`와 `결과 폴더`를 지정
- 하위 폴더 포함 / 사진 / 영상 / 여러 인물 동시 분류 옵션을 바로 선택
- `분석 및 자동 분류 시작` 버튼 하나로 핵심 작업 시작
- 진행률, 현재 파일, ETA를 메인 화면에서 바로 확인
- 기술적인 GPU/워커/threshold 옵션은 설정 화면으로 이동

### 2. EXE 옆 포터블 데이터 관리
기본 모드는 PhotoRefSorter.exe가 있는 폴더를 기준으로 아래 구조를 자동 생성합니다.

```text
PhotoRefSorter.exe
data/
  PhotoRefSorter/
    library.sqlite3
    settings.json
    thumbs/
    models/
    ...분석 캐시와 작업 기록...
```

- DB / 얼굴 임베딩 캐시 / 썸네일 / 설정 / 작업 기록을 한 위치에서 관리
- 폴더 전체를 다른 PC나 드라이브로 옮기기 쉬운 구조
- EXE 폴더에 쓰기 권한이 없으면 자동으로 사용자 프로필 저장소로 fallback
- 기존 `%LOCALAPPDATA%\PhotoRefSorter\` 데이터가 있고 새 포터블 DB가 비어 있으면 최초 실행 시 자동 이전

### 3. 저장 위치 모드
설정 → `저장 위치`에서 선택할 수 있습니다.

- `포터블` — EXE 옆 `data` 폴더 사용
- `사용자 프로필` — `%LOCALAPPDATA%` 사용
- `사용자 지정` — 원하는 데이터 루트 지정

저장 위치 변경은 다음 실행부터 적용됩니다.

### 4. 설정 화면 확장

#### 일반
- 다크 / 라이트 테마
- 마지막 사용 폴더 기억
- 기본 결과 폴더 이름

#### 분석 / 인식
- 사진 분석
- 영상 분석
- RAW 포함
- 하위 폴더 포함
- 여러 인물 동시 분류
- 기본 얼굴 인식 기준값

#### 성능
- CPU / 저장장치 자동 튜닝
- 수동 워커 수
- DirectML GPU 가속
- NTFS 하드링크 분류
- 기존 분석 캐시 재사용 여부

#### 저장 위치
- 포터블 / 사용자 프로필 / 사용자 지정
- 현재 데이터 경로 표시
- 현재 데이터 폴더 열기

#### 유지보수
- 존재하지 않는 원본 캐시 정리
- 전체 분석 / 썸네일 캐시 비우기
- 마지막 분류 작업 되돌리기
- 설정 JSON 내보내기 / 가져오기

## 기본 사용 흐름

1. `인물`에서 이름과 레퍼런스 사진을 등록합니다.
2. 첫 화면 `분석 시작`에서 사진/영상 폴더를 지정합니다.
3. PhotoRefSorter가 해당 폴더와 하위 폴더를 읽습니다.
4. 각 사진/영상에서 얼굴을 검출합니다.
5. 등록된 인물 레퍼런스와 비교합니다.
6. 인식된 파일을 인물별 결과 폴더로 분류합니다.
7. 애매한 얼굴은 `검토`, 모르는 얼굴은 `미확인 얼굴`에서 정리합니다.
8. 같은 폴더를 다시 분석하면 변경되지 않은 파일은 기존 얼굴 임베딩 캐시를 재사용합니다.

## 대량 라이브러리 최적화
- `os.scandir` 기반 빠른 폴더 열거
- 대형 JPEG 축소 디코딩
- SSD/HDD/CPU 기반 워커 자동 튜닝
- 최대 8개 병렬 얼굴 분석 워커
- SQLite WAL 얼굴 임베딩 캐시
- 변경되지 않은 사진/영상은 재분석 생략
- pHash 완전 중복 사진은 기존 얼굴 분석 결과 재사용
- NTFS 하드링크 결과 생성 옵션
- 수만 장 갤러리는 가상화 + 비동기 썸네일 방식으로 렌더링

## 인식 및 학습
- 여러 positive 레퍼런스
- negative sample 학습
- 레퍼런스 품질 가중치
- 인물별 자동 threshold
- Top 3 후보 저장
- 1위/2위 margin으로 닮은 인물 오분류 억제
- 미확인 얼굴 자동 그룹화
- 검토 결과를 다음 분류에 재학습

## 지원 미디어
- JPG / JPEG / PNG / BMP / WebP / TIFF
- HEIC / HEIF
- AVIF
- GIF
- RAW: DNG / CR2 / CR3 / NEF / ARW / ORF / RW2 / RAF / PEF / SRW
- 영상: MP4 / MOV / M4V / AVI / MKV / WebM

## v0.7 자동 검증
GitHub Actions의 Windows 환경에서 매 빌드마다 다음을 검사합니다.

1. 전체 Python 문법 / import 검사
2. HEIC / AVIF / RAW / DirectML 런타임 검사
3. v0.6 인식 / negative / pHash / 클러스터 핵심 테스트
4. v0.7 포터블 경로 계산 테스트
5. settings.json 저장/복원 테스트
6. Folder-First UI 오프스크린 smoke test
7. PyInstaller 단일 EXE 빌드
8. 패키징된 EXE 얼굴 엔진 self-test
9. EXE 옆 `data\PhotoRefSorter` 생성 확인
10. 패키징된 Folder-First UI smoke test
11. SHA256 생성 및 artifact 업로드

## 개인정보
얼굴 특징값, 썸네일, 분류 캐시는 로컬 PC에만 저장합니다. 기본 포터블 모드에서는 EXE 옆 `data\PhotoRefSorter\`에 저장하며 별도의 얼굴 인식 서버로 업로드하지 않습니다.
