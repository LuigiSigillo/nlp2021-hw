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

from transformers import DistilBertTokenizerFast,BertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import DistilBertForTokenClassification,BertForTokenClassification
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
    # model_state_dict = torch.load("model/TASKB_model_b=False_f1_48.7000.pt", map_location=torch.device(device))
    # model = StudentModelB(device = device)
    # model.load_state_dict(model_state_dict)
    modelB = StudentModelB.load_from_checkpoint("model/taskB.ckpt",device=device)
    modelB.eval()
    return modelB


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
    # model_state_dict = torch.load("model/model_A=False_f1_71.4000.pt", map_location=torch.device(device))
    # model = StudentModelAB.from_pretrained('bert-base-cased', state_dict=model_state_dict, num_labels=3)
    # model.to(device)

    model = StudentModelAB.load_from_checkpoint("model/taskA_71.4.ckpt",device = device)
    model.eval()
    
    logging.error("loadato")
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
    # model = StudentModelCD(n_classes=20,device = device)
    # model.load_state_dict(torch.load("model/TASKCD_model_b=True_f1_0.7456.pt",map_location=device))
    # model.eval()
    model = StudentModelCD.load_from_checkpoint("model/taskCD.ckpt",n_classes = 20,device = device)
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




def reconstruct_original_logits(text,logits,tokenizer):
    offset = tokenizer(text, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)['offset_mapping']
    new_logits = list()
    for i, tup in enumerate(offset):
        if tup[0] == 0:
            new_logits.append(logits[i])
    return new_logits

class PreprocessAB():
    def __init__(self, sentences):
        self.texts, _ = self.load_data(sentences)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.encodings = self.tokenizer(self.texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        self.tag2id = {'O': 0, 'B': 1, 'I': 2}
        self.id2tag = {0: 'O', 1: 'B', 2: 'I'}
        
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


class StudentModelAB(Model,pl.LightningModule):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
        
    def __init__(self, device,  comments=""):
        super(StudentModelAB,self).__init__()
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3, output_hidden_states = True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)        
        return outputs
    
    def predict(self,samples: List[Dict]) -> List[Dict]:
        targets = []
        prep = PreprocessAB(samples)
        bt_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        try:
            logging.error(self.modelB.parameters)
        except:
            logging.error("loading modelb")
            self.modelB = StudentModelB.load_from_checkpoint("model/taskB.ckpt",device=self.device)
            self.modelB.eval()
       
        for i,encode in tqdm(enumerate(prep.encodings['input_ids'])):
            json_pred = {"targets":[]}
            lst,pred = [],[]
            with torch.no_grad():
                ids = torch.unsqueeze(torch.tensor(encode),0)
                attention_mask = torch.unsqueeze(torch.tensor(prep.encodings['attention_mask'][i]),0)
                
                try:
                    logits = self.forward(torch.tensor(ids).to(self.device), torch.tensor(attention_mask).to(self.device))["logits"]
                except Exception as e:
                    #print(encode,prep.texts[i])
                    logging.error("model error", e)
                    targets.append(json_pred)
                    continue
            for id in logits[0].argmax(1):
                lst.append(id.item())
            p = reconstruct_original_logits(prep.texts[i], lst,prep.tokenizer)[1:-1]
            idtotag = [prep.id2tag[raw_pred] for raw_pred in p]
            idx_tgt_list = []
            for j,word in enumerate(idtotag):
                if word == "B":
                    json_pred["targets"].append((prep.texts[i][j],"positive"))
                    start = self.return_indices(prep.texts[i],j)
                    idx_tgt_list.append([start, start + len(prep.texts[i][j])])
                elif (word == "I") and (idtotag[j-1] == "B"):
                    try:
                        last_tuple = json_pred['targets'][-1]
                        words_tagged = last_tuple[0] + " " + prep.texts[i][j]
                        sent_tagged = last_tuple[1]
                        json_pred['targets'][-1] = (words_tagged, sent_tagged)
                        
                        idx_tgt_list[-1][1] = idx_tgt_list[-1][1] + len(prep.texts[i][j]) +1
                    except:
                        words_tagged = prep.texts[i][j]
                        sent_tagged = "positive"
                        json_pred['targets'].append((words_tagged, sent_tagged))
                        start = self.return_indices(prep.texts[i],j)
                        idx_tgt_list.append([start, start + len(prep.texts[i][j])])

                elif word == "I":
                    json_pred["targets"].append((prep.texts[i][j],"positive"))
                    start = self.return_indices(prep.texts[i],j)
                    idx_tgt_list.append([start, start + len(prep.texts[i][j])])
            batches = self.create_B_batches(prep.texts[i],json_pred,idx_tgt_list,bt_tokenizer)
            json_pred_sent = self.predict_B_after_A_2(json_pred,batches)
            targets.append(json_pred_sent)
        return targets

    def predict_B_after_A_2(self,json_pred,batches):
        cont = len(json_pred['targets'])

        id2tag = {0: 'NONE', 1: 'conflict', 2: 'negative', 3: 'neutral', 4: 'positive'}

        sentiments = []
        i = 0
        while cont > 0:
            batch = batches[i]
            input_ids = torch.unsqueeze(batch['input_ids'],0).to(self.device)
            attention_mask = torch.unsqueeze(batch['attention_mask'],0).to(self.device)
            indices_keyword = torch.unsqueeze(batch['indices'],0).to(self.device)

            with torch.no_grad():
                out = self.modelB.forward(input_ids, attention_mask=attention_mask, indices_keyword=indices_keyword)
                logits = out['logits']
                #forward_result = task_b_classifier(inputs.cpu(), idx_start.cpu())

            y_pred_labels = torch.argmax(logits, axis=-1)
            pred_label = y_pred_labels.tolist()[0]
            y_pred = id2tag[pred_label]
            sentiments.append(y_pred)
            cont-=1
            i+=1
        for k,targ in enumerate(json_pred['targets']):
            try:
                if sentiments[k] != "NONE":
                    json_pred["targets"][k] = (json_pred["targets"][k][0],sentiments[k])
            except:
                json_pred["targets"][k] = (json_pred["targets"][k][0],"conflict")

        return json_pred


    def return_indices(self,frase_splitt,word_stop):
        c = 0
        for i,w in enumerate(frase_splitt):
            if word_stop == i:
                return c
            c+=len(" "+w)
        return c
    
    def find_indices(self,new_sent):
        splitted = new_sent.split(" ")
        indices = [i+1 for i,w in enumerate(splitted) if (w=="START") or (w=="END")]
        indices[1] = indices[1]-1
        return indices


    def create_B_batches(self,text,targ_list,idx_list,bt_tokenizer):
        #id2tag = {0: 'NONE', 1: 'conflict', 2: 'negative', 3: 'neutral', 4: 'positive'}
        #tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        #encodings = self.tokenizer(self.sentences, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
        data_store = []
        text =  " ".join(text)
        sentences = []
        lst = []
        for i,(targ,_) in enumerate(targ_list['targets']):
            new_sent = text[:idx_list[i][0]]+" <START> " + text[idx_list[i][0]:idx_list[i][1]] + " <END>" + text[idx_list[i][1]:]
            new_sent = [lemmatizer.lemmatize(w)  for w in new_sent.split(" ")]
            new_sent = " ".join(self.remove_stopwords(" ".join(new_sent)))    
            index = self.find_indices(new_sent)
                
            sentences.append(new_sent)
            data_store.append((new_sent,torch.tensor(index,dtype=torch.long)))

        if len(targ_list['targets']) == 0:
            new_sent = " ".join(text)
            # new_sent = [lemmatizer.lemmatize(w)  for w in new_sent.split(" ")]
            # new_sent = " ".join(self.remove_stopwords(" ".join(new_sent)))
            index = [0,0]
            sentences.append(new_sent)
        
            data_store.append((new_sent,torch.tensor(index,dtype=torch.long)))
        
        encodings = bt_tokenizer(sentences, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
        for idx,batch in enumerate(data_store):
            item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
            item['indices'] = batch[1]
            lst.append(item)
        
        return lst



    def get_batch(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['indices'] = self.data_store[idx][1]
        return item

    def remove_stopwords(self,sent: str) -> str:
        stop_words = set(stopwords.words('english'))

        # remove punkt
        others = "–" +"—" + "−" + "’" + "”" + "“" #These chars arent inside the standard punctuation
        str_punkt = string.punctuation+ others
        translator = str.maketrans(str_punkt, ' '*len(str_punkt)) 
        word_tokens = word_tokenize(sent.translate(translator)) 
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return filtered_sentence

class PreprocessB():
    def __init__(self, sentences):
        self.data_store,self.sentences,self.targets = self.load_data(sentences)
        #self.texts, self.tags,self.data_store = self.load_data(sentences)

        self.id2tag = {0: 'NONE', 1: 'conflict', 2: 'negative', 3: 'neutral', 4: 'positive'}


        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.encodings = self.tokenizer(self.sentences, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
    
    def remove_stopwords(self,sent: str) -> str:
        stop_words = set(stopwords.words('english'))

        # remove punkt
        others = "–" +"—" + "−" + "’" + "”" + "“" #These chars arent inside the standard punctuation
        str_punkt = string.punctuation+ others
        translator = str.maketrans(str_punkt, ' '*len(str_punkt)) 
        word_tokens = word_tokenize(sent.translate(translator)) 
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return filtered_sentence
            
    def load_data(self,list_of_sentences):
        data_store,sentences,targets = [],[],[]
        for obj in list_of_sentences:
            _sentence = []
            obj['targets'] = sorted(obj['targets'], key=lambda x: x[0][0])

            for i,targ_obj in enumerate(obj['targets']):
                #print(targ_obj)
                new_sent = obj['text'][:targ_obj[0][0]-1]+" <START> " + obj['text'][targ_obj[0][0]:targ_obj[0][1]] + " <END>" + obj['text'][targ_obj[0][1]:]
                new_sent = [lemmatizer.lemmatize(w)  for w in new_sent.split(" ")]
                new_sent = " ".join(self.remove_stopwords(" ".join(new_sent)))    
                index = self.find_indices(new_sent)
                    
                sentences.append(new_sent)
                targets.append([(targ[1], "") for j,targ in enumerate(obj['targets'])])

                data_store.append((new_sent,torch.tensor(index,dtype=torch.long)))

            if len(obj['targets'])==0:
                targets.append([(targ[1], "") for j,targ in enumerate(obj['targets'])])

                new_sent = obj['text']
                # new_sent = [lemmatizer.lemmatize(w)  for w in new_sent.split(" ")]
                # new_sent = " ".join(self.remove_stopwords(" ".join(new_sent)))
                index = [0,0]
                sentences.append(new_sent)
                data_store.append((new_sent,torch.tensor(index,dtype=torch.long)))
        
        return data_store,sentences,targets
    

    def find_indices(self,new_sent):
        splitted = new_sent.split(" ")
        indices = [i+1 for i,w in enumerate(splitted) if (w=="START") or (w=="END")]
        indices[1] = indices[1]-1
        return indices


    def get_batch(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['indices'] = self.data_store[idx][1]
        return item
    

    def __len__(self) -> int:
        return len(self.sentences)


class StudentModelB(Model,pl.LightningModule):

    def __init__(self, device,  comments=""):
        super(StudentModelB,self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', num_labels=5, output_hidden_states = True)
        self.lin1 = torch.nn.Linear(768, 768)
        self.classifier = torch.nn.Linear(768, 5)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.to(device)

    def forward(self, input_ids, attention_mask, indices_keyword, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)# labels=labels)
        # hidden_states = outputs['hidden_states']
        # hidden_states = self.tuple_of_tensors_to_tensor(hidden_states)
        hidden_states = outputs['last_hidden_state']
        batch_size, seq_len, hidden_size = hidden_states.shape

        #sequence of batch x seq_len vectors 
        flat_output = hidden_states.reshape(-1, hidden_size)
        
        # start offsets of each element in the batch
        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len
        
        summary_vectors_indices_sent1 = self.get_indices_keyword(indices_keyword, sequences_offsets,0)
        summary_vectors_indices_sent2 = self.get_indices_keyword(indices_keyword, sequences_offsets,1)
        
        # we retrieve the vector of the corrseponding states for the keyword given for each sentence.
        
        summary_vectors_sent1 = flat_output[summary_vectors_indices_sent1]
        summary_vectors_sent2 = flat_output[summary_vectors_indices_sent2]
        
        # do the multiplication of these two vectors retrieved
        summary_vectors = summary_vectors_sent1 * summary_vectors_sent2
        out = self.lin1(summary_vectors)
        out = F.leaky_relu(out)
        
        logits = self.classifier(out)
        res = {}
        res['logits'] = logits
        if labels is not None:
            labels = torch.stack([labels[i][1] for i in range(labels.shape[0])])
            pred = torch.argmax(logits, -1)
            loss = self.loss_fn(logits, torch.tensor(labels) )
            res['loss'] = loss
        return res
      
    def predict(self,samples: List[Dict]) -> List[Dict]:
        targets = []
        prep = PreprocessB(samples)
        i = 0
        print(len(prep.data_store))
        while i < len(prep.data_store):
            cont = len(prep.targets[i])
            json_pred = {"targets":prep.targets[i]}
            if cont==0:
                i+=1
            sentiments = []
            while cont > 0:
                #inputs,idx_start = rnn_collate_fn([prep.data_store[i]]) # inputs in the batch
                batch = prep.get_batch(i)
                input_ids = torch.unsqueeze(batch['input_ids'],0).to(self.device)
                attention_mask = torch.unsqueeze(batch['attention_mask'],0).to(self.device)
                indices_keyword = torch.unsqueeze(batch['indices'],0).to(self.device)

                with torch.no_grad():
                    out = self.forward(input_ids, attention_mask=attention_mask, indices_keyword=indices_keyword)
                    logits = out['logits']
                    #forward_result = task_b_classifier(inputs.cpu(), idx_start.cpu())

                y_pred_labels = torch.argmax(logits, axis=-1)
                y_pred_labels.tolist()[0]
                y_pred = prep.id2tag[y_pred_labels.tolist()[0]]
                sentiments += [y_pred]
                cont-=1
                i+=1
            for k,targ in enumerate(json_pred['targets']):
                try:
                    if sentiments[k] != "NONE":
                        json_pred["targets"][k] = (json_pred["targets"][k][0],sentiments[k])
                except:
                    json_pred["targets"][k] = (json_pred["targets"][k][0],"conflict")
            targets.append(json_pred)
        return targets
    
    
    '''
    return the corresponding position of the indices of the keywords, for the sent_num passed, so the first if 0 is passed and the second if 1 is passed
    summary  = [   0,   57,  114,  171,  228, ...] 
    indices_keywords = [ [ 6, 21],[ 4, 22],[ 6, 21],[ 4, 22], ...]
    '''
    def get_indices_keyword(self,indices_keywords: Sequence[tuple], summary: Sequence[int] ,sent_num: int) -> torch.Tensor:
        tens_idx = torch.tensor([item[sent_num] for item in indices_keywords]).to(self.device)
        return tens_idx + summary



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
    def __init__(self, n_classes=20, steps_per_epoch=None, n_epochs=3, lr=2e-5,device="cpu" ):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased',num_labels=n_classes)
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

            y_pred_labels = self.classify(pred_out,0.2)
            
            y_pred = prep.mlb.inverse_transform(np.array(y_pred_labels))
            for pred in y_pred:
                try:
                    asd = pred[0].split("-")
                    json_pred["categories"].append((asd[0],asd[1]))
                except:
                    continue

            targets.append(json_pred)

        return targets





