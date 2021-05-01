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

WORDS_LIMIT = 300_000
WE_LENGTH = 50



def build_model(device: str):
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    model = StudentModel(device)

    state_dict = torch.load("model/diff_leakyrelu_0.1drop_194hidden_0.0001lr_40batch_2lstmLayer_5clipGrad_epoch_5_acc_0.6600.pt", map_location=torch.device(device))
    # load params
    model.load_state_dict(state_dict)

    model.eval()
    return model
    #return RandomBaseline()


class GloVEEmbedding():

    def __init__(self):
        self.word_vectors = dict()

    def get_word_vectors(self):
        with open('./model/glove.6B.'+str(WE_LENGTH)+'d.txt') as f:
            next(f)  # skip header
            for i, line in tqdm(enumerate(f), total=WORDS_LIMIT):
                if i == WORDS_LIMIT:
                    break
                word, *vector = line.strip().split(' ')
                vector = torch.tensor([float(c) for c in vector])
                
                self.word_vectors[word] = vector

        self.word_vectors["UNK"] = torch.tensor(np.random.random(int(WE_LENGTH)),dtype=torch.float)

        self.word_vectors["SEP"] = torch.tensor(np.random.random(int(WE_LENGTH)),dtype=torch.float)
        return self.word_vectors



class StudentModel(Model,nn.Module):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(
        self,
        device: str,
        n_hidden=194,
        drop_prob= 0.3,
        bidir=True,
    ) -> None:
        super().__init__()


        self.glove_embed = GloVEEmbedding()
        self.word_vectors = self.glove_embed.get_word_vectors()
        self.word_index, self.vectors_store = self.create_vocabulary(self.word_vectors)

        # embedding layer
        #self.embedding = torch.nn.Embedding(len(self.vectors_store),WE_LENGTH)#.from_pretrained(self.vectors_store)
        self.embedding = torch.nn.Embedding.from_pretrained(self.vectors_store)

        self.n_hidden = n_hidden
        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=self.vectors_store.size(1), hidden_size=n_hidden, num_layers=2, batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(drop_prob)

        # classification 
        if bidir:
            n_hidden = n_hidden*2
        self.lin1 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear_output = torch.nn.Linear(n_hidden, 1)

        # criterion
        self.loss_fn = torch.nn.BCELoss()
        self.device = device

    def create_vocabulary(self,word_vectors):
        word_index = dict()
        vectors_store = []

        # pad token, index = 0
        vectors_store.append(torch.rand(int(WE_LENGTH)))
        #sep token
        vectors_store.append(torch.rand(int(WE_LENGTH)))

        # unk token, index = 1
        vectors_store.append(torch.rand(int(WE_LENGTH)))

        for word, vector in word_vectors.items():
            word_index[word] = len(vectors_store)
            vectors_store.append(vector)

        word_index = defaultdict(lambda: 1, word_index)  # default dict returns 1 (unk token) when unknown word
        vectors_store = torch.stack(vectors_store)
        return word_index,vectors_store

    def sentence2indices(self,sentence: str) -> torch.Tensor:
        return torch.tensor([self.word_index[word] for word in sentence.split(' ')], dtype=torch.long)

    def forward(
        self, 
        X: torch.Tensor, 
        indices_keyword: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        
        embedding_out = self.embedding(X)
        
        recurrent_out = self.rnn(embedding_out)[0]
        
        batch_size, seq_len, hidden_size = recurrent_out.shape

        flattened_out = recurrent_out.reshape(-1, hidden_size)

        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len
        
        
        summary_vectors_indices_sent1 = self.get_indices_keyword(indices_keyword, sequences_offsets,0)
        
        summary_vectors_indices_end_first_sent = self.get_indices_keyword(indices_keyword, sequences_offsets,1)

        summary_vectors_indices_sent2 = self.get_indices_keyword(indices_keyword, sequences_offsets,2)
        
        
        summary_vectors_sent1 = flattened_out[summary_vectors_indices_sent1]
        summary_vectors_sent2 = flattened_out[summary_vectors_indices_sent2]
        
        summary_vectors = summary_vectors_sent1 - summary_vectors_sent2
        
        out = self.lin1(summary_vectors)
        out = F.leaky_relu(out)
        

        logits = self.linear_output(out).squeeze(1)
        
        pred = torch.sigmoid(logits)

        result = {'logits': logits, 'pred': pred} 
        
        return result
        
    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    def get_indices_keyword(self,indices_keywords,summary,sent_num):
        #[   0,   57,  114,  171,  228] = summary
        #[ [ 6, 21],[ 4, 22],[ 6, 21],[ 4, 22] ] = indices_keywords
        logging.error(indices_keywords)
        logging.error(sent_num)
        #tens_idx = torch.tensor([item[sent_num] for item in indices_keywords]).to(self.device)
        tens_idx = torch.tensor(indices_keywords[sent_num]).to(self.device)
        return tens_idx + summary


########################### PREPROCESSING ####################################à
    def preprocess(self,json_list,phrase2vector):
        vectors = []
        indices_list = []
        for single_json in json_list:

            keyword = single_json['sentence1'][int(single_json['start1']):int(single_json['end1'])]
            keyword2 = single_json['sentence2'][int(single_json['start2']):int(single_json['end2'])]
            lemma = single_json['lemma']

            sep = " SEP "
            lemmatized1 = self.use_only_lemma(single_json['sentence1'],lemma,keyword)
            lemmatized2 = self.use_only_lemma(single_json['sentence2'],lemma,keyword2)

            sentence =  self.remove_stopwords(lemmatized1) + sep + self.remove_stopwords(lemmatized2)
            sentence = self.handle_digits(sentence)

            indices = self.get_kwd_indices(sentence,[keyword,keyword2,lemma])
            
            vector = phrase2vector(sentence)
            
            if vector is None:
                logging.error(sentence)
                vector = torch.tensor(np.random.random(int(WE_LENGTH)),dtype=torch.float)
            while len(indices)<3:
                logging.error(sentence)
                indices.append(5)
            vectors.append(vector)
            indices_list.append(indices)
            #return self.rnn_collate_fn([(vector,indices)])
        return vectors, indices_list

    def handle_digits(self,sent):
            filtered_sentence = [w if w.isalpha() else "number" for w in sent.split(" ") ]
            return " ".join(filtered_sentence)    
    
    def remove_stopwords(self,sent):
        stop_words = set(stopwords.words('english'))
        try:
            stop_words.remove(lemma)
        except:
            pass
        # remove punkt
        others = "–" +"—" + "−" + "’" + "”" + "“"
        str_punkt = string.punctuation+ others
        translator = str.maketrans(str_punkt, ' '*len(str_punkt)) 
        word_tokens = word_tokenize(sent.translate(translator)) 
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)

    def use_only_lemma(self,sent,lemma, keyword):
        filtered_sentence = [w if not w == keyword else lemma for w in sent.split(" ") ]
        return " ".join(filtered_sentence)

    
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

    def rnn_collate_fn(self, data_elements: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] # list of (x, y,z) pairs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        X = [de[0] for de in data_elements]  # list of index tensors
        
        # to implement the many-to-one strategy
        
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)  #  shape (batch_size x max_seq_len)
        
        keyword_position = [de[1] for de in data_elements] # list of tuples indices where keyword is [[1st sent, 2nd sent]]
        keyword_position = torch.tensor(keyword_position)
        


        return X, keyword_position

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        predictions = []
        preprocessed_sample, kwds_pos = self.preprocess(sentence_pairs,self.sentence2indices)
        for i, phrase in enumerate(preprocessed_sample):
            X = torch.unsqueeze(phrase, 0).to(self.device)
            keyword_position = kwds_pos[i]
            forward_out = self(X, keyword_position)  
            #predictions.append((forward_out['pred']>0.5).float().cpu())
            predictions.append('True' if forward_out['pred'] > 0.5 else 'False')
        #return ["True" if p == torch.tensor(1, dtype=float) else "False" for p in predictions]        
        return predictions

