import numpy as np
from typing import *
import json
import string
# torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from model import Model
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import logging
nltk.download('stopwords')
nltk.download('punkt')

torch.manual_seed(42)
np.random.seed(42)

WORDS_LIMIT = 400_000
WE_LENGTH = 50

class GloVEEmbedding():

    def __init__(self):
        self.word_vectors = dict()


    def get_word_vectors(self):
        with open('model/glove.6B.'+str(WE_LENGTH)+'d.txt', encoding='utf8') as f:
            for i, line in tqdm(enumerate(f), total=WORDS_LIMIT):
                if i == WORDS_LIMIT:
                    break
                word, *vector = line.strip().split(' ')
                vector = torch.tensor([float(c) for c in vector])
                
                self.word_vectors[word] = vector

        self.word_vectors["UNK"] = torch.rand(int(WE_LENGTH))

        self.word_vectors["SEP"] = torch.rand(int(WE_LENGTH))
        return self.word_vectors


def create_vocabulary(word_vectors):
    word_index = dict()
    vectors_store = []

    # pad token, index = 0
    vectors_store.append(torch.rand(int(WE_LENGTH)))
    # unk token, index = 1
    vectors_store.append(word_vectors["UNK"])

    # sep token, index = 2
    vectors_store.append(word_vectors["SEP"])

    for word, vector in word_vectors.items():
        word_index[word] = len(vectors_store)
        vectors_store.append(vector)

    word_index = defaultdict(lambda: 1, word_index)  # default dict returns 1 (unk token) when unknown word
    vectors_store = torch.stack(vectors_store)
    return word_index,vectors_store


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    glove_embed = GloVEEmbedding()
    word_vectors = glove_embed.get_word_vectors()
    word_index, vectors_store = create_vocabulary(word_vectors)
    
    n_hidden, drop_prob, bidir, n_layer_lstm = 82, 0.25, True, 2
    model = StudentModel(device,vectors_store, word_index, n_hidden, drop_prob,bidir, n_layer_lstm)
    
    model.load_state_dict(torch.load("model/diff_leakyrelu_0.2drop_82hidden_0.0001lr_40batch_2lstmLayer_1clipGrad_epoch_15_acc_0.6740.pt", map_location=torch.device(device)))
    
    model.eval()

    return model




class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(nn.Module):
    
    def __init__(
        self,
        device: str,
        vectors_store: torch.Tensor,
        word_index,
        n_hidden: int,
        drop_prob: float,
        bidir: bool,
        n_layer_lstm: int
    ) -> None:
        super().__init__()

        self.word_index = word_index
        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(vectors_store)
        self.n_hidden = n_hidden
        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=vectors_store.size(1), hidden_size=n_hidden, num_layers=n_layer_lstm, batch_first=True, bidirectional=bidir)

        # classification 
        if bidir:
           n_hidden = n_hidden*2
        self.lin1 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear_output = torch.nn.Linear(n_hidden, 1)

        self.loss_fn = torch.nn.BCELoss()
        self.device = device
        


    def forward(
        self, 
        X: torch.Tensor, 
        indices_keyword: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        
        embedding_out = self.embedding(X)
        # recurrent encoding
        recurrent_out = self.rnn(embedding_out)[0]
        # here we utilize the sequences length to retrieve the last token 
        # output for each sequence
        
        batch_size, seq_len, hidden_size = recurrent_out.shape

        # we flatten the recurrent output now I have a long sequence of batch x seq_len vectors 
        flattened_out = recurrent_out.reshape(-1, hidden_size)
        
        # tensor of the start offsets of each element in the batch
        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len
        
        summary_vectors_indices_sent1 = self.get_indices_keyword(indices_keyword, sequences_offsets,0)
        
        #summary_vectors_indices_end_first_sent = self.get_indices_keyword(indices_keyword, sequences_offsets,1)

        summary_vectors_indices_sent2 = self.get_indices_keyword(indices_keyword, sequences_offsets,2)
        

        # we retrieve the vecttor of the corrseponding states for the keyword given for each sentence.
          
        summary_vectors_sent1 = flattened_out[summary_vectors_indices_sent1]
        summary_vectors_sent2 = flattened_out[summary_vectors_indices_sent2]
        
        # do the difference of these two vectors yet retrieved
        summary_vectors = summary_vectors_sent1 * summary_vectors_sent2
        
        # feedforward pass on the summary
        out = self.lin1(summary_vectors)
        out = F.leaky_relu(out)
        

        logits = self.linear_output(out).squeeze(1)
        
        pred = torch.sigmoid(logits)

        result = {'logits': logits, 'pred': pred} 
        
        # compute loss
        if y is not None:
            loss = self.loss(pred, y)
            result['loss'] = loss
        
        return result
        
       
    def loss(self, pred, y):
        return self.loss_fn(pred, y)
    '''
    return the corresponding position of the indices of the keywords, for the sent_num passed, so the first if 0 is passed and the second if 2 is passed

    '''
    def get_indices_keyword(self,indices_keywords: Sequence[tuple], summary: Sequence[int] ,sent_num: int) -> torch.Tensor:
        #[   0,   57,  114,  171,  228] = summary
        #[ [ 6, 21],[ 4, 22],[ 6, 21],[ 4, 22] ] = indices_keywords
        #tens_idx = torch.tensor([item[sent_num] for item in indices_keywords]).to(self.device)
        tens_idx = torch.tensor(indices_keywords[sent_num]).to(self.device)
        return tens_idx + summary

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        predictions = []
        preprocessed_sample, kwds_pos = self.init_structure(sentence_pairs)
        for i, phrase in enumerate(preprocessed_sample):
            X = torch.unsqueeze(phrase, 0).to(self.device)
            keyword_position = kwds_pos[i]
            forward_out = self(X, keyword_position)  
            #predictions.append((forward_out['pred']>0.5).float().cpu())
            predictions.append('True' if forward_out['pred'] > 0.5 else 'False')
        #return ["True" if p == torch.tensor(1, dtype=float) else "False" for p in predictions]        
        return predictions


    def init_structure(self, f):
        data_store_2 = []
        data_store = []

        USE_SEP = True
        lemmatization = True
        LOWERED = False
        remove_stop_words = True
        for single_json in f:
            keyword = single_json['sentence1'][int(single_json['start1']):int(single_json['end1'])]
            keyword2 = single_json['sentence2'][int(single_json['start2']):int(single_json['end2'])]
            lemma = single_json['lemma']

            sep = " " if not USE_SEP else " SEP "
            
            if lemmatization:
                lemmatized1 = self.use_only_lemma(single_json['sentence1'],lemma,keyword)
                lemmatized2 = self.use_only_lemma(single_json['sentence2'],lemma,keyword2)
            else:
                lemmatized1 = single_json['sentence1']
                lemmatized2 = single_json['sentence2']
            
            if LOWERED:
                lemmatized1 = lemmatized1.lower()
                lemmatized2 = lemmatized2.lower()
                keyword = keyword.lower()
                keyword2 = keyword2.lower()
                lemma = lemma.lower()
            
            if remove_stop_words:
                lemmatized1_without_stop = self.remove_stopwords(lemmatized1,lemma)
                lemmatized2_without_stop = self.remove_stopwords(lemmatized2,lemma)
                sentence =  lemmatized1_without_stop + sep + lemmatized2_without_stop
                # substitue digits with "number"
                sentence = self.handle_digits(sentence)
            else:
                sentence = lemmatized1 + sep + lemmatized2

            if USE_SEP:
                indices = self.get_kwd_indices(sentence,[keyword,keyword2,lemma])
            else:
                indices = self.get_kwd_indices(lemmatized1_without_stop,[keyword,keyword2,lemma]) + [42] +self.get_kwd_indices(lemmatized2_without_stop,[keyword,keyword2,lemma])


            vector = torch.tensor([self.word_index[word] for word in sentence.split(' ')], dtype=torch.long)
            
            if vector is None or len(indices)!=3:
                print(sentence,indices, keyword) 
                continue
                
            data_store.append(vector)
            data_store_2.append(indices)
        return data_store,data_store_2

    '''
    Substitute every digits with the word "number"
    '''
    def handle_digits(self,sent: str) -> str:
        filtered_sentence = [w if w.isalpha() else "number" for w in sent.split(" ") ]
        return " ".join(filtered_sentence)       

    '''
    Removing the stopwords and the punctuation but there is the possibility that the keyword is contained
    inside the set of stopwords so I remove it first.
    '''
    def remove_stopwords(self,sent: str,lemma: str) -> str:
        stop_words = set(stopwords.words('english'))
        try:
            stop_words.remove(lemma)
        except:
            pass

        # remove punkt
        others = "–" +"—" + "−" + "’" + "”" + "“" #These chars arent inside the standard punctuation
        str_punkt = string.punctuation+ others
        translator = str.maketrans(str_punkt, ' '*len(str_punkt)) 
        word_tokens = word_tokenize(sent.translate(translator)) 
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)

    '''
    Lemmatization of the sentence
    '''
    def use_only_lemma(self,sent:str ,lemma: str, keyword: str) -> str:
        filtered_sentence = [w if not w == keyword else lemma for w in sent.split(" ") ]
        return " ".join(filtered_sentence)

    '''
    The indices of the keyword are retrieved, we have to handle the fact that the keywords can be repeated inside the sentence.
    So if we are using the separator we use this word SEP to separate the sentences and retrieve the correct indices.
    The list returned must contain three elements if we are using the separator: 
     - index of first occurence of keyword in first sentence
     - index of sep
     - index of first occurence of keyword in second sentence
    '''
    def get_kwd_indices(self,sentence: str,keywords: Sequence[str]) -> Sequence[int]:
        i = 0
        j_list = []
        sec = False
        sentence_list = sentence.split(" ")
        while i < len(sentence_list):
            if sentence_list[i] == "SEP":
                sec = True
                j_list.append(i)
            if sentence_list[i] in keywords:
                if j_list == []:
                    j_list.append(i)
                elif sec:
                    j_list.append(i)
                    return j_list
            i += 1
        return j_list
            
            