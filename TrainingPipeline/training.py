from __future__ import absolute_import, division, print_function, unicode_literals
from metaflow import FlowSpec,Parameter, step, batch, retry,catch,S3
import pandas as pd
import random
import numpy as np
import os
import torch
import tensorflow as tf
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import boto3
import logging
import smart_open

class Train(FlowSpec):

    @step
    def start(self):
        import torch

        # Load the dataset into a pandas dataframe.
        with S3() as s3:
            df = pd.read_csv(smart_open.smart_open('s3://sentstorage/scrape/labelledtweets.tsv'),delimiter='\t', header=None, names=['label','tweet'])
        #df = pd.read_csv("t.tsv", delimiter='\t', header=None, names=['label','tweet'])
        self.tweets = df.tweet.values
        self.labels = df.label.values

        # Load the BERT tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_len = 0

        # For every sentence...
        for self.tweet in self.tweets:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            self.input_ids = self.tokenizer.encode(self.tweet, add_special_tokens=True)

            # Update the maximum sentence length.
            self.max_len = max(self.max_len, len(self.input_ids))



        # Tokenize all of the sentences and map the tokens to thier word IDs.
        self.input_ids2 = []
        self.attention_masks2 = []

        # For every sentence...
        for self.tweet in self.tweets:
            self.encoded_dict = self.tokenizer.encode_plus(
                                self.tweet,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
            # Add the encoded sentence to the list.
            self.input_ids2.append(self.encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            self.attention_masks2.append(self.encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        self.input_ids2 = torch.cat(self.input_ids2, dim=0)
        self.attention_masks2 = torch.cat(self.attention_masks2, dim=0)
        self.labels = torch.tensor(self.labels)

        # Combine the training inputs into a TensorDataset.
        self.dataset = TensorDataset(self.input_ids2, self.attention_masks2, self.labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        self.train_size = int(0.9 * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size

        # Divide the dataset by randomly selecting samples.
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])


        # The DataLoader needs to know our batch size for training, so we specify it
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch
        # size of 16 or 32.
        self.batch_size = 32

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        self.train_dataloader = DataLoader(
                    self.train_dataset,  # The training samples.
                    sampler = RandomSampler(self.train_dataset), # Select batches randomly
                    batch_size = self.batch_size # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.validation_dataloader = DataLoader(
                    self.val_dataset, # The validation samples.
                    sampler = SequentialSampler(self.val_dataset), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                )

        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 3,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        self.optimizer = AdamW(self.model.parameters(),
                          lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

        # Number of training epochs. The BERT authors recommend between 2 and 4.
        self.epochs = 2

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        self.total_steps = len(self.train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        """self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps = 0,
                                                    num_training_steps = self.total_steps)"""


        # Set the seed value all over the place to make this reproducible.
        self.seed_val = 42

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)

        for self.epoch_i in range(0, self.epochs):

            # Reset the total loss for this epoch.
            self.total_train_loss = 0

            self.model.train()

            # For each batch of training data...
            for self.step, self.batch in enumerate(self.train_dataloader):

                self.b_input_ids = self.batch[0]
                self.b_input_mask = self.batch[1]
                self.b_labels = self.batch[2]

                self.model.zero_grad()

                self.loss, self.logits = self.model(self.b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=self.b_input_mask,
                                     labels=self.b_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                self.total_train_loss += self.loss.item()

                # Perform a backward pass to calculate the gradients.
                self.loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # Update the learning rate.
                #self.scheduler.step()


        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        torch.save(self.model_to_save.state_dict(), 'saved.pt')
        #with S3(s3root='s3://sentstorage/model') as s3:
        #    s3.put_files([('model','model')])
        #self.model_to_save.save_pretrained('s3://sentstorage/model')
        #self.tokenizer.save_pretrained('/Users/sid/Desktop/train')
        #with S3(s3root='s3://sentstorage/model') as s3:
        #    s3.put_many(self.model_to_save.save_pretrained())




        self.next(self.end)


    @catch(print_exception=False)
    @step
    def end(self):
        print("Saved model to bucket")





if __name__ == '__main__':
    Train()
