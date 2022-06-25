
# Question Answering Model

[2022년 인공지능 온라인 경진대회](https://aichallenge.or.kr/competition/detail/1)에서 자연어 영역의 문서 검색 효율화를 위한 기계독해 과제에 참여했으며, 이는
텍스트와 질문이 주어졌을 때 본문에서 질문의 답을 찾는 문제이다. PLM(Pretrained Language Model)을 활용해 본문과 질문, 질문에 대한 정답을 학습시키 딥러닝으로 하여금 문맥과 패턴 등을 학습시키고 테스트 데이터에 대해 정답을 예측하도록 한다. 정확히는 본문에서 정답이 되는 텍스트의 시작 인덱스와 종료 인덱스를 예측한다.

HuggingFace🤗의 [Question Answering Tutorial](https://huggingface.co/docs/transformers/tasks/question_answering)을 참고해 베이스라인을 작성했으며, 정답이 없는 경우를 학습시켜 테스트 데이터 결과를 제출한 결과 exact match 점수는 0.8 이상을 안정적으로 기록했다. 여기에 여러가지 한국어 PLM을 테스팅하고 최종 정답을 추출하는 샘플링 방법, 하이퍼 파라미터 테스팅을 진행하는 방식으로 접근했다. 

기계독해 문제의 어려운 점:
1. 주어진 본문에서 답을 찾아야 하는 문제로 질문이 애매하면 답을 잘 추론하지 못한다. 
2. 부분적으로 답이 맞더라고 하더라고 정확하게 답의 처음과 끝이 맞지 않으면 틀린다. (샘플링 방법 중요)
3. 본문과 유사한 표현으로 바꾸거나 우회적으로 질문하면 답을 잘 추론하지 못하는 경우가 있다.

---

## How to Use

### Install python libraries

```bash
pip install numpy==1.20.1
pip install transformers
pip install datasets
```

### Hyperparameter testing

This is a template repository for Text Classification using Optimum and onnxruntime to support generic inference with Hugging Face Hub generic Inference API. There are two required steps:

1. Specify the requirements by defining a `requirements.txt` file.
2. Implement the `pipeline.py` `__init__` and `__call__` methods. These methods are called by the Inference API. The `__init__` method should load the model and preload the optimum model and tokenizers as well as the `text-classification` pipeline needed for inference. This is only called once. The `__call__` method performs the actual inference. Make sure to follow the same input/output specifications defined in the template for the pipeline to work.

add 
```
library_name: generic
```
to the readme.

머머머 해서 결과 제출 했을 때 08.-0.87 최종 6위로 끝남
