import os
import argparse

import pandas as pd
import numpy as np
import collections

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from modules.dataset import QADataset, QADatasetValid, QADatasetTest


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--pretrained_model_name', 
                required=True,
                default='monologg/kobigbird-bert-base',
                help="Set pretrained model. (Examples: klue/bert-base, monologg/kobert, ...")

    p.add_argument('--test_file', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_answer_length', type=int, default=40)

    config = p.parse_args()

    return config


def main(config):
    test = pd.read_csv(config.test_file)
    test_dataset = QADatasetValid(test['input_ids'], test['token_type_ids'], test['attention_mask'], test['offset_mapping'], test['example_id'])
    test_set = QADatasetTest(test['input_ids'], test['token_type_ids'], test['attention_mask'])

    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(config.pretrained_model_name)
        model.load_state_dict(saved_data) 

        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device

        test_dataloader = DataLoader(test_set, collate_fn=default_data_collator, batch_size=config.batch_size, shuffle=False)

        # Don't forget turn-on evaluation mode.
        model.eval()

        # Predictions
        start_logits = []
        end_logits = []
        for batch in test_dataloader:
            x = torch.tensor(batch['input_ids']).to(device)
            token_type_ids = torch.tensor(batch['token_type_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            outputs = model(x,  token_type_ids=token_type_ids, attention_mask=attention_mask)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())
            
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(test_dataset)]
        end_logits = end_logits[: len(test_dataset)]
    
    # create answers
    n_best = 5

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(test_dataset):
        example_to_features[feature["example_id"]].append(idx)  

    predicted_answers = []
    for i in range(len(test)):
        example_id = test.loc[i]["question_id"]
        context = test.loc[i]["context"]
        answers = []
    
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]   
            end_logit = end_logits[feature_index]
            offsets = test_dataset[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()   
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index
                        or end_index - start_index + 1 > config.max_answer_length):
                        continue

                    answer = {"text": context[offsets[start_index][0] : offsets[end_index][1]],   
                            "logit_score": start_logit[start_index] + end_logit[end_index]}
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""}) 
        


if __name__ == '__main__':
    config = define_argparser()
    main(config)