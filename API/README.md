# main API 사용법

1. 하나의 경로에 다음 4가지 파일을 둡니다.
```
    model.ckpt.19
    main.py
    main_modules.py
    requirements.txt
```

- pre_trained된 koelectra모델은 다음 경로에서 다운로드 가능합니다.
    https://drive.google.com/drive/u/0/folders/1g7w50TyDS1psv2crh6kImDkHPUInAfRd

<br/>
<br/>

2. 터미널에서 가상환경을 만들고(python=3.8) 해당 경로로 이동 후, 아래의 명령어를 통해 패키지를 설치합니다. 
```
pip install -r requirements.txt
```
<br/>
<br/>

3. 아래의 명령어로 실행합니다.
```
uvicorn main:app --reload
```
<br/>
<br/>

4. 링크를 따라 http://127.0.0.1:8000/ 로컬주소에 진입합니다. 주소에 docs를 추가하여 실행할 수 있습니다.(페이지가 뜰때까지 몇초정도 시간이 소요될 수 있습니다. 새로고침을 눌러주세요)
```
http://127.0.0.1:8000/docs
```


- 주소에 docs를 추가하여 실행할 수 있습니다.(페이지가 뜰때까지 몇초정도 시간이 소요될 수 있습니다. 새로고침을 눌러주세요)

![gif1](https://user-images.githubusercontent.com/98604716/159411018-e95f9657-1113-4357-8866-d758257d1244.gif)

- POST와 Classifier 중앙에 위치한 '/'모양을 누르고 try it out 버튼을 클릭합니다.

![gif2](https://user-images.githubusercontent.com/98604716/159411026-2070b0ed-9160-4eb6-ab03-4f7895f9980a.gif)

- 'string'부분에 두개의 문장을 각각 적은 뒤 'Exercute'버튼을 누르면 문장 유사도를 확인할 수 있습니다.

![gif3](https://user-images.githubusercontent.com/98604716/159411028-5273549c-261b-4cf3-bd61-d32536501dad.gif)

