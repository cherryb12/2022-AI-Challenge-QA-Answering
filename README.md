
# Question Answering Model

[2022년 인공지능 온라인 경진대회](https://aichallenge.or.kr/competition/detail/1)에서 자연어 영역의 문서 검색 효율화를 위한 기계독해 과제에 참여했으며, 이는
텍스트와 질문이 주어졌을 때 본문에서 질문의 답을 찾는 문제이다. PLM(Pretrained Language Model)을 활용해 본문과 질문, 질문에 대한 정답을 학습시키 딥러닝으로 하여금 테스트 데이터에 대해 정답을 예측하도록 한다. 정확히는 본문에서 정답이 되는 텍스트의 시작 인덱스와 종료 인덱스를 예측한다.

HuggingFace🤗의 [Question Answering Tutorial](https://huggingface.co/docs/transformers/tasks/question_answering)을 참고해 베이스라인을 작성했으며, 정답이 없는 경우를 학습시켜 테스트 데이터 결과를 제출한 결과 exact match 점수는 0.8 이상을 안정적으로 기록했다. 여기에 여러가지 한국어 PLM을 테스팅하고 최종 정답을 추출하는 샘플링 방법, 하이퍼 파라미터 테스팅을 진행하는 방식으로 접근했다. 

기계독해 문제의 어려운 점:
1. 주어진 본문에서 답을 찾아야 하는 문제로 질문이 애매하면 답을 잘 추론하지 못한다. 
2. 부분적으로 답이 맞더라고 하더라고 정확하게 답의 처음과 끝이 맞지 않으면 틀린다. (샘플링 방법 중요)
3. 본문과 유사한 표현으로 바꾸거나 우회적으로 질문하면 답을 잘 추론하지 못하는 경우가 있다.

---

## How to Use

### Install python libraries

```bash
pip install -r requirements.txt
```

### Preprocess
train, test 테이터를 각각 전처리해 pickle 파일로 저장한다. 

```bash
python ./modules/preprocess.py --data_path --save_path --pretrained_model_name --max_length --stride
```

### Train
전처리가 끝난 데이터를 불러와 학습시킨다. 

```bash
python ./hf_trainer.py --model_fn --file_path --pretrained_model_name --n_epochs 2 --batch_size --n_best --max_answer_length
```

### Inference
학습 후 저장된 모델 가중치를 불러와 테스트 데이터를 예측한다. 

```bash
python ./inference.py --model_fn --file_path --pretrained_model_name --n_best --max_answer_length
```

### Hyperparameter tuning
|Hyperparameters||
|-|-|
|plm|klue/bert, klue/roberta, kobigbird, koelectra|
|max_length|200, 384, 512|
|stride|50, 128|
|n_epochs|2, 5|
|batch_size|16, 32|
|warmup_ratio|0.1|
|n_best|1, 5, 20|
|max_answer_length|40|

kobigbird 모델로 max_length 384, stride 50, n_epochs 2, batch_size 16, n_best 5로 테스트했을 때 validation data의 exact match 점수가 80.69였으며 해당 모델의 가중치로 테스트 데이터의 결과를 대시보드에 제출한 결과 최고 0.870까지 점수를 얻을 수 있었다. <br/>

하지만 데이터 split 시 random_state 값을 지정하지 않아 테스트 시마다 결과가 0.85-0.86 정도로 달라지는 우를 범하고 말핬다... 대회 종료 후 조정된 점수를 보니 0.8634로 최종 6위로 대회를 마쳤다. 
<br/><br/>
![대회결과](https://user-images.githubusercontent.com/96293754/175795336-89ebd945-6028-4634-a06e-26e118e2368a.png){: width="70%" height="50%"}

