import os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_path',
        required=True,
        help="Directory where data files located.")

    p.add_argument(
        "--save_path",
        required=True,
        help="Directory to save preprocessed dataset.")

    p.add_argument(
        '--test_size',
        required=True,
        default=.2,
        type=float,
        help="Set test size. Input float number")

    p.add_argument('--pretrained_model_name', 
                    required=True,
                    default='monologg/kobigbird-bert-base',
                    help="Set pretrained model. (Examples: klue/bert-base, monologg/kobert, ...")
    
    p.add_argument('--max_length', type=int, default=384)
    p.add_argument('--stride', type=int, default=50)

    config = p.parse_args()
    return config

# preprocess_training_example function
def preprocess_training_example(example):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=config.max_length,
        truncation="only_second",
        stride=config.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)

        sample_idx = sample_map[i]
        answer_start = example['answer_start']
        text = example['text']

        if len(answer_start) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

        else:
            start_char = answer_start[0]
            end_char = answer_start[0] + len(text[0])
        
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# preprocess_validation_example function
def preprocess_validation_example(example):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=config.max_length,
        truncation="only_second",
        stride=config.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        example_ids.append(example["question_id"])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [o if k==0 or sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

    inputs["example_id"] = example_ids
    return inputs


def main(config):

    datapath = config.data_path
    savepath = config.save_path

    # read data files
    if os.path.isdir(datapath):  # 경로가 존재하는 지 확인
        file_list = os.listdir(datapath)  #  해당 경로의 모든 파일 리스트
    print(f"{len(file_list)} files found : ", file_list)

    for i, file in enumerate(file_list):
        if file.endswith(".json"):
            file_list[i] = pd.read_json(os.path.join(datapath, file))
        else:
            file_list[i] = pd.read_csv(os.path.join(path, file))
        print(f"file {i} : ", file_list[i].shape[0])

    # read train json file into dataframe
    cols = ['context', 'question_id', 'question', 'answer_start', 'text']

    comp_list = []
    for row in train['data']:
        for i in range(len(row['paragraphs'])):
            if len(row['paragraphs'][i]['qas'][0]['answers']) !=0:
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question_id'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question'])
                temp_list.append([row['paragraphs'][i]['qas'][0]['answers'][0]['answer_start']])
                temp_list.append([row['paragraphs'][i]['qas'][0]['answers'][0]['text']])
                comp_list.append(temp_list)

            else:
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question_id'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['answers'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['answers'])
                comp_list.append(temp_list)

    train = pd.DataFrame(comp_list, columns=cols) 

    # read test json file into dataframe
    cols = ['context', 'question_id', 'question']

    comp_list = []
    for row in test['data']:
        for i in range(len(row['paragraphs'])):
            for j in range(len(row['paragraphs'][i]['qas'])):
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][j]['question_id'])
                temp_list.append(row['paragraphs'][i]['qas'][j]['question'])
                comp_list.append(temp_list)

    test = pd.DataFrame(comp_list, columns=cols) 

    # answer_start correction
    answer_start = []

    for i in range(len(train['answer_start'])):
        if train['text'][i]!= train['text_comparision'][i]:
            answer_start.append([train['answer_start'][i][0]+1])
        else:
            answer_start.append(train['answer_start'][i])

    train['answer_start'] = answer_start

    # train and validation split
    train, validation = train_test_split(train, config.test_size, random_state=42, shuffle=True)
    train = train.reset_index()
    validation = validation.reset_index(drop=True)

    # preprocess train data
    inputs = []
    for i in range(len(train)):
        inputs.append(preprocess_training_example(train.loc[i]))

    input_ids = []
    token_type_ids = []
    attention_mask = []
    start_positions = []
    end_positions = []

    for input in inputs:
        for i in range(len(input['input_ids'])):
            input_ids.append(input['input_ids'][i])
            token_type_ids.append(input['token_type_ids'][i])
            attention_mask.append(input['attention_mask'][i])
            start_positions.append(input['start_positions'][i])
            end_positions.append(input['end_positions'][i])

    inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'start_positions': start_positions, 'end_positions': end_positions}
    train = pd.DataFrame(inputs)


    # preprocess validation data
    inputs = []

    for i in range(len(validation)):
        inputs.append(preprocess_validation_example(validation.loc[i]))

    input_ids = []
    token_type_ids = []
    attention_mask = []
    offset_mapping = []
    example_id = []

    for input in inputs:
        for i in range(len(input['input_ids'])):
            input_ids.append(input['input_ids'][i])
            token_type_ids.append(input['token_type_ids'][i])
            attention_mask.append(input['attention_mask'][i])
            offset_mapping.append(input['offset_mapping'][i])
            example_id.append(input['example_id'][i])

    inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'offset_mapping': offset_mapping, 'example_id': example_id}
    validation = pd.DataFrame(inputs)

    # preprocess test data
    inputs = []

    for i in range(len(test)):
        inputs.append(preprocess_validation_example(test.loc[i]))

    input_ids = []
    token_type_ids = []
    attention_mask = []
    offset_mapping = []
    example_id = []

    for input in inputs:
        for i in range(len(input['input_ids'])):
            input_ids.append(input['input_ids'][i])
            token_type_ids.append(input['token_type_ids'][i])
            attention_mask.append(input['attention_mask'][i])
            offset_mapping.append(input['offset_mapping'][i])
            example_id.append(input['example_id'][i])

    inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'offset_mapping': offset_mapping, 'example_id': example_id}
    test= pd.DataFrame(inputs)

    # save files
    train.to_csv(os.path.join(savepath, 'preprocessed_train.csv'), index=False)
    validation.to_csv(os.path.join(savepath, 'preprocessed_validation.csv'), index=False)
    test.to_csv(os.path.join(savepath, 'preprocessed_test.csv'), index=False)


if __name__ == "__main__":
    config = define_argparser()
    main(config)