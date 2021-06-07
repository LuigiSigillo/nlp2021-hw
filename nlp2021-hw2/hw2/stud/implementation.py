import numpy as np
from typing import List, Tuple, Dict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from model import Model
import random
import nltk
import logging
from typing import *
from pprint import pprint
import string

from torchtext import data
from torchtext.vocab import Vectors
from torchtext.vocab import *



import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import DistilBertForTokenClassification
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertModel,get_linear_schedule_with_warmup


torch.manual_seed(42)
np.random.seed(42)


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    model_state_dict = torch.load("model/check_model_b=True_f1_0.5406.pt", map_location=torch.device(device))
    model = StudentModelB.from_pretrained('distilbert-base-cased', state_dict=model_state_dict, num_labels=5)
    model.to(device)
    model.eval()
    return model


def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    model_state_dict = torch.load("model/model_b=False_f1_0.8386.pt", map_location=torch.device(device))
    model = StudentModelAB.from_pretrained('distilbert-base-cased', state_dict=model_state_dict, num_labels=3)
    model.to(device)
    model.eval()
    
    
    return model


def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    model = StudentModelCD(n_classes=20,device = device)
    model.load_state_dict(torch.load("model/TASKCD_model_b=True_f1_0.7456.pt",map_location=device))
    model.eval()

    return model

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds




def new_logits(text,logits,tokenizer):
    offset = tokenizer(text, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)['offset_mapping']
    new_logits = list()
    for i, tup in enumerate(offset):
        if tup[0] == 0:
            new_logits.append(logits[i])
    return new_logits


class PreprocessAB():
    def __init__(self, sentences):
        self.texts, _ = self.load_data(sentences)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        self.encodings = self.tokenizer(self.texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        
        self.tag2id = {'O':0, 'I':1, 'B':2}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}
        
        #self.labels = self.encode_tags(self.tags, self.encodings)
        

    def load_data(self,list_of_sentences):
        sentences,texts,tags = [], [], []
        for obj in list_of_sentences:
            _sentence = []
            for t in tokenizer.tokenize(obj['text']):
                token = {"token": t}
                _sentence.append(token)
            sentences.append(_sentence)

        for elem in sentences:
            texts.append([tok['token'] for tok in elem])
            #tags.append([tag['ne_label'] for tag in elem])
        return texts, tags
    
    
    def encode_tags(self,tags, encodings):
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)
            try:
                # set labels whose first offset position is 0 and the second is not 0
                doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
                encoded_labels.append(doc_enc_labels.tolist())
            except:
                print(doc_labels, doc_offset)

        return encoded_labels


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class StudentModelAB(Model,DistilBertForTokenClassification):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
        

    def predict(self,samples: List[Dict]) -> List[Dict]:
        targets = []
        prep = PreprocessAB(samples)
        try:
            logging.error(self.modelB.parameters)
        except:
            logging.error("SI")
            device = "cuda" if next(self.parameters()).is_cuda else "cpu"
            model_state_dictB = torch.load("model/check_model_b=True_f1_0.5406.pt", map_location=torch.device(device))
            self.modelB = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', state_dict=model_state_dictB, num_labels=5)
            self.modelB.eval()
       
        for i,encode in tqdm(enumerate(prep.encodings['input_ids'])):
            json_pred = {"targets":[]}
            lst,pred = [],[]
            ids = torch.unsqueeze(torch.tensor(encode),0)
            attention_mask = torch.unsqueeze(torch.tensor(prep.encodings['attention_mask'][i]),0)
            logits = self(torch.tensor(ids).to("cpu"), torch.tensor(attention_mask).cpu())["logits"]
            
            for id in logits[0].argmax(1):
                lst.append(id.item())
            p = new_logits(prep.texts[i], lst,prep.tokenizer)[1:-1]
            idtotag = [prep.id2tag[raw_pred] for raw_pred in p]

            for j,word in enumerate(idtotag):
                if word == "B":
                    json_pred["targets"].append((prep.texts[i][j],"positive"))
                elif (word == "I") and (idtotag[j-1] == "B"):
                    try:
                        last_tuple = json_pred['targets'][-1]
                        words_tagged = last_tuple[0] + " " + prep.texts[i][j]
                        sent_tagged = last_tuple[1]
                        json_pred['targets'][-1] = (words_tagged, sent_tagged)
                    except:
                        words_tagged = prep.texts[i][j]
                        sent_tagged = "positive"
                        json_pred['targets'].append((words_tagged, sent_tagged))
                elif word == "I":
                    json_pred["targets"].append((prep.texts[i][j],"positive"))

            targets.append(self.predictBAB(json_pred,prep.texts[i],prep.tokenizer,ids,attention_mask))
        return targets

    def predictBAB(self,json_pred,text,tokenizer,ids,attention_mask) -> List[Dict]:
        tag2id = {'neutral':0, 'negative':1, 'conflict':2, 'O':3,'positive':4 }
        id2tag = {id: tag for tag, id in tag2id.items()}
        
        lst,pred = [],[]
        logits = self.modelB(torch.tensor(ids).to("cpu"), torch.tensor(attention_mask).cpu())["logits"]
        for id in logits[0].argmax(1):
            lst.append(id.item())
        p = new_logits(text, lst,tokenizer)[1:-1]
        idtotag = [id2tag[raw_pred] for raw_pred in p]
        sentiments = []
        
        idd = 0
        if (len(json_pred['targets'])>0):
            for word, sentiment in zip(text,idtotag):
                if word in json_pred["targets"][idd][0]:
                    if sentiment !="O": 
                        json_pred["targets"][idd] = (json_pred["targets"][idd][0],sentiment)
                    else:
                        json_pred["targets"][idd] = (json_pred["targets"][idd][0],"positive") 
                    idd+=1
                    if idd==len(json_pred["targets"]):
                        break
        
        return json_pred

class PreprocessB():
    def __init__(self, sentences):
        self.texts, self.tags = self.load_data(sentences)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        self.encodings = self.tokenizer(self.texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        
        self.tag2id = {'neutral':0, 'negative':1, 'conflict':2, 'O':3,'positive':4 }

        self.id2tag = {id: tag for tag, id in self.tag2id.items()}
                

    def load_data(self,list_of_sentences):
        sentences,texts,tags = [], [], []
        for obj in list_of_sentences:
            _sentence = []
            for t in tokenizer.tokenize(obj['text']):
                ne_label ="O"
                for i in range(len(obj['targets'])):
                    if t in obj['targets'][i][1]: #if target word
                        ne_label = obj['targets'][i][1]
                        break
                        
                token = {"token": t, "ne_label": ne_label}
                _sentence.append(token)

            sentences.append(_sentence)

        for i,elem in enumerate(sentences):
            texts.append([tok['token'] for tok in elem])
            #new_lst = list()
            tags.append([(targ[1], "") for j,targ in enumerate(list_of_sentences[i]['targets'])])
        return texts, tags
    

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class StudentModelB(Model,DistilBertForTokenClassification):
    '''
    --> !!! STUDENT: implement here your predict function !!! <--
    Args:
        - If you are doing model_b (ie. aspect sentiment analysis):
            sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                [
                    {
                        "text": "I love their pasta but I hate their Ananas Pizza.",
                        "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                    },
                    {
                        "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        "targets": [[4, 9], "people", [[36, 40], "taste"]]
                    }
                ]
        - If you are doing model_ab or model_cd:
            sentence: a dictionary that represents an input sentence, for example:
                [
                    {
                        "text": "I love their pasta but I hate their Ananas Pizza."
                    },
                    {
                        "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                    }
                ]
    Returns:
        A List of dictionaries with your predictions:
            - If you are doing target word identification + target polarity classification:
                [
                    {
                        "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                    },
                    {
                        "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                    }
                ]
            - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                [
                    {
                        "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                        "categories": [("food", "conflict")]
                    },
                    {
                        "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                        "categories": [("service", "positive"), ("food", "positive")]
                    }
                ]
    ''' 
      
    def predict(self,samples: List[Dict]) -> List[Dict]:
        targets = []
        prep = PreprocessB(samples)
        for i,encode in enumerate(prep.encodings['input_ids']):
            json_pred = {"targets":prep.tags[i]}
            lst,pred = [],[]
            ids = torch.unsqueeze(torch.tensor(encode),0)
            attention_mask = torch.unsqueeze(torch.tensor(prep.encodings['attention_mask'][i]),0)

            logits = self(torch.tensor(ids).to("cpu"), torch.tensor(attention_mask).cpu())["logits"]

            for idx in logits[0].argmax(1):
                lst.append(idx.item())
            p = new_logits(prep.texts[i], lst,prep.tokenizer)[1:-1]
            idtotag = [prep.id2tag[raw_pred] for raw_pred in p]
            sentiments = []
            
            idd = 0
            if (len(json_pred['targets'])>0):
                for word, sentiment in zip(prep.texts[i],idtotag):
                    if word in json_pred["targets"][idd][0]:
                        if sentiment !="O": 
                            json_pred["targets"][idd] = (json_pred["targets"][idd][0],sentiment)
                        else:
                            json_pred["targets"][idd] = (json_pred["targets"][idd][0],"conflict") 
                        idd+=1
                        if idd==len(json_pred["targets"]):
                            break
            targets.append(json_pred)
        return targets



class PreprocessCD():
    def __init__(self, sentences):
        self.texts = self.load_data(sentences)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        self.max_len = 200
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit_transform([['ambience-conflict', 'ambience-negative', 'ambience-neutral',
       'ambience-positive', 'anecdotes/miscellaneous-conflict',
       'anecdotes/miscellaneous-negative',
       'anecdotes/miscellaneous-neutral',
       'anecdotes/miscellaneous-positive', 'food-conflict',
       'food-negative', 'food-neutral', 'food-positive', 'price-conflict',
       'price-negative', 'price-neutral', 'price-positive',
       'service-conflict', 'service-negative', 'service-neutral',
       'service-positive']])
        self.encodings = self.encode_texts()
        
    def encode_texts(self):
        encodings = []
        for item_idx,text in enumerate(self.texts):
            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True, # Add [CLS] [SEP]
                max_length= self.max_len,
                padding = 'max_length',
                return_token_type_ids= False,
                return_attention_mask= True, # Differentiates padded vs normal token
                truncation=True, # Truncate data beyond max length
                return_tensors = 'pt' # PyTorch Tensor format
            )
            
            input_ids = inputs['input_ids'].flatten()
            attn_mask = inputs['attention_mask'].flatten()
            
            encodings.append({
                'input_ids': input_ids ,
                'attention_mask': attn_mask,
            })
        return encodings
        

    def load_data(self,list_of_sentences):
        sentences = []     
        texts = []

        for obj in list_of_sentences:
            texts.append(obj['text'])
            
        return texts  
    

    def __len__(self):
        return len(self.labels)


class StudentModelCD(Model,pl.LightningModule):
    def __init__(self, n_classes=5, steps_per_epoch=None, n_epochs=3, lr=2e-5,device="cuda" ):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased',num_labels=n_classes)
        #self.classifier = nn.Linear(self.bert.config.hidden_size,n_classes) # outputs = number of labels
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.to(device)
    
    def forward(self,input_ids, attn_mask):
        output = self.bert(input_ids = input_ids ,attention_mask = attn_mask)
        #pooler output --> Last layer hidden-state of the first token of the sequence (classification token) further processed 
        #by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next 
        #sentence prediction (classification) objective during pretraining.
        #output = self.classifier(output.pooler_output)
        output = output['logits']        
        return output
    
    '''
    --> !!! STUDENT: implement here your predict function !!! <--
    Args:
        - If you are doing model_b (ie. aspect sentiment analysis):
            sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                [
                    {
                        "text": "I love their pasta but I hate their Ananas Pizza.",
                        "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                    },
                    {
                        "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        "targets": [[4, 9], "people", [[36, 40], "taste"]]
                    }
                ]
        - If you are doing model_ab or model_cd:
            sentence: a dictionary that represents an input sentence, for example:
                [
                    {
                        "text": "I love their pasta but I hate their Ananas Pizza."
                    },
                    {
                        "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                    }
                ]
    Returns:
        A List of dictionaries with your predictions:
            - If you are doing target word identification + target polarity classification:
                [
                    {
                        "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                    },
                    {
                        "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                    }
                ]
            - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                [
                    {
                        "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                        "categories": [("food", "conflict")]
                    },
                    {
                        "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                        "categories": [("service", "positive"), ("food", "positive")]
                    }
                ]
    ''' 
      
    def classify(self,pred_prob,thresh):
        y_pred = []
        for tag_label_row in pred_prob:
            temp=[]
            for tag_label in tag_label_row:
                if tag_label >= thresh:
                    temp.append(1) 
                else:
                    temp.append(0) 
            y_pred.append(temp)
        return y_pred

    def predict(self,samples: List[Dict]) -> List[Dict]:
        targets = []
        prep = PreprocessCD(samples)
        for i,batch in enumerate(prep.encodings):

            json_pred = {"targets":[], "categories": []}
            b_input_ids = batch["input_ids"]
            b_attn_mask = batch["attention_mask"]
            
            ids = torch.unsqueeze(torch.tensor(b_input_ids),0).cpu()
            attmasks = torch.unsqueeze(torch.tensor(b_attn_mask),0).cpu()
            # Forward pass, calculate logit predictions
            pred_out = self(ids,attmasks)
            pred_out = torch.sigmoid(pred_out)
            
            pred_out = pred_out.detach().cpu().numpy()

            y_pred_labels = self.classify(pred_out,0.23)
            
            y_pred = prep.mlb.inverse_transform(np.array(y_pred_labels))
            for pred in y_pred:
                try:
                    asd = pred[0].split("-")
                    json_pred["categories"].append((asd[0],asd[1]))
                except:
                    continue

            targets.append(json_pred)

        return targets





