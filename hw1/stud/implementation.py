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
nltk.download('stopwords')
nltk.download('punkt')

WORDS_LIMIT = 140_000
WE_LENGTH = 100



def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    model = StudentModel(device)
    '''glove_embed = GloVEEmbedding()
    word_vectors = glove_embed.get_word_vectors()
    word_index,vectors_store = create_vocabulary(word_vectors)
    lm = LoadModel(vectors_store)'''

    state_dict = torch.load("./model/difference_leakyrelu__epoch_24.pt", map_location=torch.device(device))
    # create new OrderedDict that does not contain `module.`
    '''from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params'''
    model.load_state_dict(state_dict)

    model.eval()
    return model
    #return RandomBaseline()


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


class GloVEEmbedding():

    def __init__(self):
        self.word_vectors = dict()

    def get_word_vectors(self):
        with open('./model/glove.6B.100d.txt') as f:
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

def create_vocabulary(word_vectors):
    word_index = dict()
    vectors_store = []

    # pad token, index = 0
    vectors_store.append(torch.rand(int(WE_LENGTH)))

    # unk token, index = 1
    vectors_store.append(torch.rand(int(WE_LENGTH)))

    for word, vector in word_vectors.items():
        word_index[word] = len(vectors_store)
        vectors_store.append(vector)

    word_index = defaultdict(lambda: 1, word_index)  # default dict returns 1 (unk token) when unknown word
    vectors_store = torch.stack(vectors_store)
    return word_index,vectors_store


glove_embed = GloVEEmbedding()
word_vectors = glove_embed.get_word_vectors()
word_index,vectors_store = create_vocabulary(word_vectors)

def sentence2indices(sentence: str) -> torch.Tensor:
    return torch.tensor([word_index[word] for word in sentence.split(' ')], dtype=torch.long)

class Preprocessing():

    def preprocess(self,json_string,phrase2vector):
        single_json = json_string

        keyword = single_json['sentence1'][int(single_json['start1']):int(single_json['end1'])]
        keyword2 = single_json['sentence2'][int(single_json['start2']):int(single_json['end2'])]
        lemma = single_json['lemma']

        sep = " SEP "
        lemmatized1 = self.use_only_lemma(single_json['sentence1'],lemma,keyword)
        lemmatized2 = self.use_only_lemma(single_json['sentence2'],lemma,keyword2)

        sentence =  self.remove_stopwords(lemmatized1) + sep + self.remove_stopwords(lemmatized2)
        
        indices = self.get_kwd_indices(sentence,[keyword,keyword2,lemma])
        
        vector = phrase2vector(sentence)
        
        if vector is None or len(indices)!=3:
            #print(sentence,indices)
            return None
            
        return self.rnn_collate_fn([(vector,indices)])

    def remove_stopwords(self,sent):
        stop_words = set(stopwords.words('english'))
        # remove punkt
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
        word_tokens = word_tokenize(sent.translate(translator)) 
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)

    def use_only_lemma(self,sent,lemma, keyword):
        filtered_sentence = [w if not w == keyword else lemma for w in sent.split(" ") ]
        return " ".join(filtered_sentence)

    
    def get_kwd_indices(self,sentence,keywords):
        i = 0
        j = []
        sec = False
        sentence_list = sentence.split(" ")
        while i < len(sentence_list):
            if sentence_list[i] == "SEP":
                sec = True
                j.append(i)
            if sentence_list[i] in keywords:
                if j == []:
                    j.append(i)
                elif sec:
                    j.append(i)
                    return j
            i += 1
        return j

    def rnn_collate_fn(self,data_elements: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        X = [de[0] for de in data_elements]  # list of index tensors
        # to implement the many-to-one strategy
        X_lengths = torch.tensor([x.size(0) for x in X], dtype=torch.long)
        

        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)  #  shape (batch_size x max_seq_len)

        keyword_position = [de[1] for de in data_elements] # list of tuples indices where keyword is [[1st sent, 2nd sent]]


        keyword_position = torch.tensor(keyword_position)

        return X, X_lengths, keyword_position

class StudentModel(Model, nn.Module):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(
        self,
        device: str,
        n_hidden=128,
        drop_prob= 0.3,
        bidir=True,
    ) -> None:
        super().__init__()



        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(vectors_store)
        self.n_hidden = n_hidden
        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=vectors_store.size(1), hidden_size=n_hidden, num_layers=1, batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(drop_prob)

        # classification 
        if bidir:
            n_hidden = n_hidden*2
        self.lin1 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear_output = torch.nn.Linear(n_hidden, 1)

        # criterion
        self.loss_fn = torch.nn.BCELoss()
        self.device = device



    def forward(
        self, 
        X: torch.Tensor, 
        X_length: torch.Tensor,
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
        
        # and we use a simple trick to compute a tensor of the indices of the last token in each batch element
        #last_word_relative_indices = X_length - 1
        # tensor of the start offsets of each element in the batch
        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len
        # e.g. (0, 5, 10, 15, ) + ( 3, 2, 1, 4 ) = ( 3, 7, 11, 19 )
        #summary_vectors_indices = sequences_offsets + last_word_relative_indices
        
        
        summary_vectors_indices_sent1 = self.get_indices_keyword(indices_keyword, sequences_offsets,0)
        
        summary_vectors_indices_end_first_sent = self.get_indices_keyword(indices_keyword, sequences_offsets,1)

        summary_vectors_indices_sent2 = self.get_indices_keyword(indices_keyword, sequences_offsets,2)
        

        # finaly we retrieve the vectors that should summarize every sentence.
        # (i.e. the last token in the sequence)
        #summary_vectors = flattened_out[summary_vectors_indices]
        
        summary_vectors_sent1 = flattened_out[summary_vectors_indices_sent1]
        summary_vectors_sent2 = flattened_out[summary_vectors_indices_sent2]
        
        #summary_vectors = torch.mean(torch.stack((summary_vectors_sent1,summary_vectors_sent2)), dim = 0)
        summary_vectors = summary_vectors_sent1 * summary_vectors_sent2
        
        # now we can classify the sentences with a feedforward pass on the summary
        # vectors
        out = self.lin1(summary_vectors)
        out = F.leaky_relu(out)
        

        # compute logits (which are simply the out variable) and the actual probability distribution (pred, as it is the predicted distribution)
        logits = self.linear_output(out).squeeze(1)
        
        pred = torch.sigmoid(logits)

        result = {'logits': logits, 'pred': pred} #'hidden':hidden}
        
        # compute loss
        if y is not None:
            loss = self.loss(pred, y)
            result['loss'] = loss
        
        return result
        
    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    def get_indices_keyword(self,indices_keywords,summary,sent_num):
        #[   0,   57,  114,  171,  228] = summary
        #[ [ 6, 21],[ 4, 22],[ 6, 21],[ 4, 22] ] = indices_keywords
        tens_idx = torch.tensor([item[sent_num] for item in indices_keywords]).to(self.device)
        return tens_idx + summary

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        predictions = []
        prep = Preprocessing()
        for phrase in sentence_pairs:
            X, X_lengths, keyword_position = prep.preprocess(phrase,sentence2indices)
            forward_out = self(X, X_lengths, keyword_position)  # add a dimension to create a one-item batch
            '''for i,prob in enumerate(forward_out["pred"]):
                print("\n {}".format( prob) )'''
            predictions.append((forward_out['pred']>0.5).float().cpu())
        return ["True" if p == torch.tensor(1, dtype=float) else "False" for p in predictions]        

