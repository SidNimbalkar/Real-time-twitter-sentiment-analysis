import pandas as pd
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf

def predict(tweet):
    model = BertForSequenceClassification.from_pretrained("model")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_idstest = []
    attention_maskstest = []

    encoded_dicttest = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=200, pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt', )

    input_idstest.append(encoded_dicttest['input_ids'])

    attention_maskstest.append(encoded_dicttest['attention_mask'])

    # Convert the lists into tensors.
    input_idstest = torch.cat(input_idstest, dim=0)
    attention_maskstest = torch.cat(attention_maskstest, dim=0)

    outputs = model(input_idstest, token_type_ids=None, attention_mask=attention_maskstest)
    out = outputs[0]
    sc = torch.max(out[0])
    score = sc.data.numpy()
    b = torch.argmax(out[0])
    idx = np.unravel_index(b, out[0].shape)
    sentiment = idx[0]

    return(score,sentiment)


scores,sentiments = predict("I am happy")

print(scores)
print(sentiments)
