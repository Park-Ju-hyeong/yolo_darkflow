
물체인식 알고리즘중에서 yolo 를 이용한 방식을 정리합니다. (이론 x, 돌리는법 o)

순서는

1. YOLO 소개  
2. dataset, 자동차 번호판   
3. annotation, label 만드는 작업  
4. darkflow, yolo for tensorflow  
5. training, 학습  
6. detection, test 하기  

## [You Only Look Once][YOLO]

YOLO, object detection - 물체인식  
찾으려는 물체를 찾아 `bounding box` 로 표시해주는 알고리즘 입니다.  
아래 사진은 YOLO 를 설명하는 대표적인 개, 자전거를 인식한 결과입니다.   

YOLO 와 비슷한 선상에 있는 알고리즘으로는 `Fast/Faster R-CNN` 과 `SSD` 가 있는데 이것들과의 차이점으로는 다음과 같은 특징이 있습니다.  

|속도 : speed |정학도 : accuracy|
|:--:|:--:|
|`YOLO` > <br> `Fast/Faster R-CNN`, `SSD`| `Fast/Faster R-CNN`, `SSD` > <br> `YOLO` |  

각자 필요에 맞게, 상황에 맞게 알고리즘을 가져다 쓰면 됩니다.  
real-time (예를 들어, gpu 1장 기준 초당 30장 이상 처리要 : `30fps` ) 을 해야하는 상황이면 YOLO 를 쓰면 되고  
시간과 비용에 구애받지 않는다면, Faster R-CNN 을 쓰면 됩니다.  

## [dataset][DATASET]

이번 포스트를 위해서 자동차 번호판 인식을 진행해보려 하는데, google 에 `자동차 번호판` 을 검색해서 나온 사진들을 이용하였습니다.  (29장 구할 수 있었습니다.)  

자동차 번호판은 `[0-9]`숫자와 `[가-힣]`한글의 조합으로 구성되어있습니다.  
이번 포스트에서 인식한 클레스는 숫자 10종류와 한글 1종류로 제한하겠습니다.  
숫자는 `MNIST` 처럼 얕은 네트워크도 0~9 를 잘 인식하지만, 한글은 종류가 너무 많아서 통으로 `kr` 이라는 class로 정의했습니다.

## Annotation

Object detection 는 지도학습에 해당돼서 당연히 정답 데이터가 필요합니다.  
여기서 정답데이터의 형태는 우리가 일반적으로 생각하는 [[1],[2],[3],[4], ...] 이런 형태가 아니라 해당 오브젝트의 좌표값이 필요합니다.  
(1024*1024 이미지에서 시작좌표:(300, 400) 와 끝좌표 : (500, 600) 이런식으로)  
또한 구현된 여러 코드들을 살펴보면 xml 로 정답 데이터를 요구하는 경우가 대부분입니다.  
이런 데이터셋을 만드려면 처음에는 <S>바로 포기하거나</S> 어떻게 해야하는 지 모를 것입니다.  

여기서 annotation - xml 데이터를 만들어주는 중요한 프로그램이 있는데 바로 `labelimg` 입니다.  

### [labelimg][labelimg]

```
git clone https://github.com/tzutalin/labelImg.git  

cd labelImg-master  

python labelImg.py
```

다음 코드를 통해서 `labelImg`를 실행시킵니다.  
실행 조건에는 pyqt가 필요한데 아마 anconda를 통해 python을 사용하시는 분들이라면 아무 문제없이 작동될 것입니다.  

1. 이후 open 말고 `opendir`을 통해서 이미지가 들어있는 폴더를 선택해서 안에있는 이미지 전체를 불러옵니다.  
2. `savedir` 을 통해서 annotation 정보가 담겨있는 xml 파일이 저장될 폴더를 선택합니다.  
3. 열심히 노가다 합니다.  


작업에 도움되는 단축키는 이거 3개면 충분합니다.  

`w`를 눌러서 내가 원하는 물체를 감싸주고 label 정보를 입력해주는 작업을 반복합니다.  
`d`를 눌러 다음 이미지로 넘어갑니다.  
`a`를 눌러 이전 이미지로 돌어갑니다.  

| 단축키 | 설명 |
|:--:|:--:|
| `w`	| Create a rect box |
| `d`	| Next image |
| `a`	| Previous image |  

## [darkflow][darkflow]

`darkflow` 는 `[darknet]` 의 tnesorflow 버전입니다.  (darknet은 YOLO를 C로 짠 오픈소스입니다.)  
[Darknet: Open Source Neural Networks in C][dn]  

멋진 분들이 짜주신 오픈소스입니다.   
우리는 감사히 사용하고 [GNU General Public License v3.0][gnu]`Liability, Warranty`만 지켜주시면 됩니다.  

아직 pip 등록된 패키지가 아니므로 git 으로 설치해야합니다.   

```
git clone https://github.com/thtrieu/darkflow.git

cd darkflow

python setup.py build_ext --inplace

pip install .
```

이제 터미널에서 `flow` 라는 명령어로 `yolo v1` 과 `yolo v2` 를 사용할 수  있습니다.  
현재 공식적으로 `yolo v3`가 나온 상황이지만 darknet 에서만 지원되고 darkflow 에서는 사용할 수 없습니다. [Issues #665][665]   

클론한 darkflow에는 이미 네트워크를 구성해 놓았는데,  

| cfg | 설명 |
|:--:|:--:|
| `yolo.cfg`	| 가장 베이직한 모델, VGG 기반 수정버전 |
| `tiny-yolo.cfg`	| 말 그대로 가벼운 모델, AlexNet 기반 수정버전 |
| `~~coco.cfg`	| coco data set 에 맞춰서 네트워크 변경 |  
| `~~voc.cfg`	| voc data set 에 맞춰서 네트워크 변경 |  

이번 포스트에서는 yolo.cfg 를 사용할 것이다.  아래는 네트워크 구조  

|Source | Train? | Layer description                | Output size|
|:-----:|:------:|:---------------------------------|:--------------|
|       |        | input                            | (?, 608, 608, 3)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 608, 608, 32)|
| Load  |  Yep!  | maxp 2x2p0_2                     | (?, 304, 304, 32)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 304, 304, 64)|
| Load  |  Yep!  | maxp 2x2p0_2                     | (?, 152, 152, 64)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 152, 152, 128)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 152, 152, 64)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 152, 152, 128)|
| Load  |  Yep!  | maxp 2x2p0_2                     | (?, 76, 76, 128)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 76, 76, 256)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 76, 76, 128)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 76, 76, 256)|
| Load  |  Yep!  | maxp 2x2p0_2                     | (?, 38, 38, 256)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 256)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 256)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 38, 38, 512)|
| Load  |  Yep!  | maxp 2x2p0_2                     | (?, 19, 19, 512)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 19, 19, 512)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 19, 19, 512)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Load  |  Yep!  | concat [16]                      | (?, 38, 38, 512)|
| Init  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 38, 38, 64)|
| Load  |  Yep!  | local flatten 2x2                | (?, 19, 19, 256)|
| Load  |  Yep!  | concat [27, 24]                  | (?, 19, 19, 1280)|
| Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 19, 19, 1024)|
| Init  |  Yep!  | conv 1x1p0_1    linear           | (?, 19, 19, 80)|


사실 `coco` 나 `voc` 데이터 셋을 사용하는게 아니라 자신의 데이터 셋을 사용하는 것이라면,  
2가지 정도 수정할 사항이 있다.  

### 1. labels.txt 수정하기  

`./labels.txt`  을 위에서 `labelimg`에서 만들때 썼던 classes 이름으로 수정해 준다.  

```
0
1
2
3
4
5
6
7
8
9
kr
```

### 2. .cfg 수정하기  

사용할 `yolo.cfg` 를 복사해서 다른이름으로 저장하고  
(ex. car-yolo.cfg : 왜냐하면 그 이름 그대로 쓰면 안에 로직으로 인해 coco label로 인식해버린다. labels.txt 소용이 없어짐)  

  + `[region]` classes 를 11로 바꾼다.
  + 그 위 `[convolutional]` filters 값을 (anchors + classes) * num 로 바꿔준다. 여기서는 `(5 + 11) * 5 = 80`  
    - anchors 는 2개가 1세트이다. 즉 10개 아니다. 표현방법이 이상할 뿐...

```
[convolutional]
size=1
stride=1
pad=1
filters=80 # 수정!
activation=linear


[region]
anchors =  0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
bias_match=1
classes=11 # 수정!
coords=4
num=5
softmax=1
jitter=.3
rescore=1
```

### flow --help : parameters

```
Example usage: flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/yolo.weights

Arguments:
  --help, --h, -h  show this super helpful message and exit
  --imgdir         path to testing directory with images
  --binary         path to .weights directory
  --config         path to .cfg directory
  --dataset        path to dataset directory
  --labels         path to labels file
  --backup         path to backup folder
  --summary        path to TensorBoard summaries directory
  --annotation     path to annotation directory
  --threshold      detection threshold
  --model          configuration of choice
  --trainer        training algorithm
  --momentum       applicable for rmsprop and momentum optimizers
  --verbalise      say out loud while building graph
  --train          train the whole net
  --load           how to initialize the net? Either from .weights or a checkpoint, or even from scratch
  --savepb         save net and weight to a .pb file
  --gpu            how much gpu (from 0.0 to 1.0)
  --gpuName        GPU device name
  --lr             learning rate
  --keep           Number of most recent training results to save
  --batch          batch size
  --epoch          number of epoch
  --save           save checkpoint every ? training examples
  --demo           demo on webcam
  --queue          process demo in batch
  --json           Outputs bounding box information in json format.
  --saveVideo      Records video from input video or camera
  --pbLoad         path to .pb protobuf file (metaLoad must also be specified)
  --metaLoad       path to .meta file generated during --savepb that corresponds to .pb file
```

### Training  

1. `../data/dataset/ ` 경로에 바로 이미지를 넣고 (.png, .jpg)
2. `../data/annotations/ \` 경로에 바로 annotation 데이터를 넣습니다.  (.xml)
3. trainer 는 기본 `rmsprop` 이지만 `Adam`으로 바꿔줬다. - 개인취향  
4. 그리고 학습할 때는 꼭 `--train` 을 해야 합니다.
5. --load 는 맨 처음 학습할 때는 없애고, 재학습 할 때만 `-1` or `특정 epoch 숫자` 를 넣습니다.  
  + .weights 파일을 보실 수 있는데 전혀 쓸일이 없습니다.  왜냐하면, coco 나 voc 데이터이기 때문입니다.  
  + 우리는 ckpt 를 쓰거나 pb 로 저장해서 씁니다.    
6. 나머지는  `parameters` 를 참조.  

```
flow \
--model ./cfg/car-yolo.cfg \
--labels ./labels.txt \
--trainer adam \
--load -1 \
--dataset ../data/dataset/ \
--annotation ../data/annotations/ \
--train \
--summary ./logs \
--batch 5 \
--epoch 100 \
--save 50 \
--keep 5 \
--lr 1e-04 \
```

여기서 조금 특이한 점은 logs 로 summary 했는데 logstrain 에 저장됩니다.  
logs는 빈 폴더가 됩니다.  

```
tensorboard --logdir=./logstrain
```

### detect  

학습이 끝나면 detection 을 진행합니다.  
`--dataset` 이 아니라 `--imgdir` 에 찾고싶은 데이터를 입력합니다.  
그러면 test할 이미지 폴더 안에 `out` 폴더가 생기고 디텍션한 결과가 저장됩니다.  

```
flow \
--imgdir ../data/dataset/ \
--model ./cfg/car-yolo.cfg \
--load -1 \
--batch 1 \
--threshold 0.5 \
```  

## 마무리

YOLO 를 돌려보고 싶은 마음에 google 을 해봤지만 죄다 이론설명..  
darkflow 깃헙 페이지를 가도 충분하지는 않다.  
그래서 만들게 됐다.  

[YOLO]:         https://pjreddie.com/darknet/yolo/  
[DATASET]:      https://www.google.co.kr/search?q=%EC%9E%90%EB%8F%99%EC%B0%A8+%EB%B2%88%ED%98%B8%ED%8C%90&safe=off&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjkz5u9iajaAhVJtpQKHfnVDe0Q_AUICigB&biw=1440&bih=900  
[labelimg]:     https://github.com/tzutalin/labelImg  
[darkflow]:     https://github.com/thtrieu/darkflow
[dn]:           https://pjreddie.com/darknet/
[gnu]:          https://github.com/thtrieu/darkflow/blob/master/LICENSE
[665]:          https://github.com/thtrieu/darkflow/issues/665