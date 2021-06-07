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

lemmatizer = WordNetLemmatizer()


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
    
    window_size, window_shift = 100, 100
    vocabulary = torch.load('model/vocab_obj.pth')
    hparams = {
        'vocab_size': 4427,#len(vocabulary),
        'hidden_dim': 128,
        'embedding_dim': 100,
        'num_classes': 4, # number of different universal POS tags
        'bidirectional': True,
        'num_layers': 1,
        'dropout': 0.0
    }
    
    model = StudentModel(hparams, device, vocabulary)
    model.load_state_dict(torch.load("model/epoch_25_f1_0.6692.pt", map_location=torch.device(device)))
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

   
    window_size, window_shift = 100, 100
    vocabulary = torch.load('model/vocab_obj.pth')
    hparams = {
        'vocab_size': 4427,#len(vocabulary),
        'hidden_dim': 128,
        'embedding_dim': 100,
        'num_classes': 4, # number of different universal POS tags
        'bidirectional': True,
        'num_layers': 1,
        'dropout': 0.0
    }
    
    model = StudentModel(hparams, device, vocabulary)
    model.load_state_dict(torch.load("model/epoch_25_f1_0.6692.pt", map_location=torch.device(device)))
    model.eval()
    logging.error("CIAO")
    return model

    # return RandomBaseline(mode='ab')

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
    # return RandomBaseline(mode='cd')
    raise NotImplementedError

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



class Preprocess():
    def __init__(self, window_size:int, window_shift:int=-1,device="cpu"):
        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        self.encoded_data = None
        self.device = device
    def remove_stopwords(self,sent: str) -> str:
            stop_words = set(stopwords.words('english'))

            # remove punkt
            others = "–" +"—" + "−" + "’" + "”" + "“" #These chars arent inside the standard punctuation
            str_punkt = string.punctuation+ others
            translator = str.maketrans(str_punkt, ' '*len(str_punkt)) 
            word_tokens = word_tokenize(sent.translate(translator)) 
            
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            return filtered_sentence
    def preprocess(self, list_of_sentences):
        sentences = []
        for obj in list_of_sentences:
            _sentence = []
            lemmatized = [lemmatizer.lemmatize(w)  for w in obj['text'].split(" ")]
            lemmatized = self.remove_stopwords(" ".join(lemmatized))
            for t in lemmatized:
                token = {"token": t}
                _sentence.append(token)
            sentences.append(_sentence)
        return sentences
    
    def create_windows(self, sentences): # to save space in memory ? why we do not load directly all the sentence?
        """ 
        Args:
            sentences (list of lists of dictionaries, where each dictionary represents a word occurrence)
        """
        data = []
        for sentence in sentences:
            for i in range(0, len(sentence), self.window_shift):
                window = sentence[i:i+self.window_size]
                if len(window) < self.window_size:
                    window = window + [None]*(self.window_size - len(window))  # to match the same length of sentences
                assert len(window) == self.window_size
                data.append(window)
        self.data = data
        return data


    def index_dataset(self, l_vocabulary):
        self.encoded_data = list()
        for i in range(len(self.data)):
            # for each window
            sentence = self.data[i]
            encoded_sentence = torch.LongTensor(self.encode_text(sentence, l_vocabulary)).to(self.device)
            
            # for each element d in the elem window (d is a dictionary with the various fields from the CoNLL line) 
            #encoded_labels = torch.LongTensor([l_label_vocabulary[d["ne_label"]] if d is not None else l_label_vocabulary["<pad>"] for d in sentence]).to(self.device)
            
            self.encoded_data.append({"inputs":encoded_sentence }) #"outputs":encoded_labels})
        return self.encoded_data
    
    @staticmethod
    def encode_text(sentence:list, l_vocabulary):
        """
        Args:
            sentences (list): list of OrderedDict, each carrying the information about one token.
            l_vocabulary (Vocab): vocabulary with mappings from words to indices and viceversa.
        Return:
            The method returns a list of indices corresponding to the input tokens.
        """
        indices = list()
        for w in sentence:
            if w is None:
                indices.append(l_vocabulary["<pad>"])
            elif w["token"] in l_vocabulary.stoi: # vocabulary string to integer (necessary to search faster)
                indices.append(l_vocabulary[w["token"]])
            else:
                indices.append(l_vocabulary["<unk>"])
        return indices
    
    @staticmethod
    def decode_output(outputs:torch.Tensor,l_label_vocabulary: Vocab={'<pad>': 0, 'B': 2, 'I': 3, 'O': 1}):
        """
        Args:
            outputs (Tensor): a Tensor with shape (batch_size, max_len, label_vocab_size)
                containing the logits outputed by the neural network.
            l_label_vocabulary (Vocab): is the vocabulary containing the mapping from
            a string label to its corresponding index and vice versa
        Output:
            The method returns a list of batch_size length where each element is a list
            of labels, one for each input token.
        """
        l_label_vocabulary = torch.load("model/vocab_obj_label.pth")

        max_indices = torch.argmax(outputs, -1).tolist() # shape = (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            # vocabulary integer to string is used to obtain the corresponding word from the max index
            predictions.append([l_label_vocabulary.itos[i] for i in indices])
        return predictions
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        return self.encoded_data[idx]
    
    def get_raw_element(self, idx):
        return self.data[idx]

class StudentModel(Model, pl.LightningModule):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    
    def __init__(self, hparams, device, vocabulary, embeddings = None, *args, **kwargs):
        super(StudentModel, self).__init__(*args, **kwargs)
        """
        Args:
            model: the model we want to train.
            hparams: hyperparams defined by pl module
        """
        self.vocabulary = vocabulary
        #self.save_hyperparameters(hparams)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.model = POSTaggerModel(hparams, embeddings)
        #self.device = device
    # This performs a forward pass of the model, as well as returning the predicted index.
    def forward(self, x):
        logits = self.model(x)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    # This runs the model in training mode mode, ie. activates dropout and gradient computation. It defines a single training step.
    def training_step(self, batch, batch_nb):
        '''
        {
        'inputs': tensor([  5, 121,  34,   6, 834,  68, 307,   4, 370, 684, 663,  40,  42, 748, 0,   0,   0,   0,   0], device='cuda:0'), 
        'outputs': tensor([1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
        }
        '''
        inputs = batch['inputs']
        labels = batch['outputs']
        # We receive one batch of data and perform a forward pass:
        logits, _ = self.forward(inputs)
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        # Compute the loss:
        loss = self.loss_function(logits, labels)
        # Log it:
        self.log('train_loss', loss, prog_bar=True)
        # Very important for PL to return the loss that will be used to update the weights:
        return loss

    # This runs the model in eval mode, ie. sets dropout to 0 and deactivates grad. Needed when we are in inference mode.
    def validation_step(self, batch, batch_nb):
        inputs = batch['inputs']
        labels = batch['outputs']

        logits, _ = self.forward(inputs)
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        sample_loss = self.loss_function(logits, labels)
        self.log('valid_loss', sample_loss, prog_bar=True)

    
    def predict(self, samples: List[Dict]) -> List[Dict]:
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
        prep = Preprocess(100,100)
        preprocessed_samples = prep.preprocess(samples)
        prep.create_windows(preprocessed_samples)
        encoded_data = prep.index_dataset(l_vocabulary=self.vocabulary)
        
        #logging.error(encoded_data)
        #logging.error(preprocessed_samples)
        predictions_list = []
        for i,x in enumerate(encoded_data):
            prediction_single = {"targets" : []}
            test_x = x["inputs"].to("cpu")
            logits, predictions = self(test_x.unsqueeze(0))
            decoded_labels = prep.decode_output(logits)[0]
            #logging.error(decoded_labels)

            #[{"targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] }] # A list having a tuple for each target word
            logging.error(preprocessed_samples[i])

            for idx, predicted_label in enumerate(decoded_labels):
                #logging.error(self.vocabulary.itos[idx]) #input
                #logging.error(predicted_label) #predcition
                try:
                    logging.error(preprocessed_samples[i][idx])
                except:
                    pass

                if predicted_label == "B":
                    prediction_single["targets"].append((preprocessed_samples[i][idx]['token'],"positive"))

        #logging.error(prediction_single)
        predictions_list.append(prediction_single)
        return predictions_list


class POSTaggerModel(nn.Module):
    def __init__(self, hparams, embeddings = None):
        super(POSTaggerModel, self).__init__()
        # Embedding layer: a matrix vocab_size x embedding_dim where each index 
        # correspond to a word in the vocabulary and the i-th row corresponds to  
        # a latent representation of the i-th word in the vocabulary.
        self.word_embedding = nn.Embedding(hparams['vocab_size'], hparams["embedding_dim"])

        if embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding = nn.Embedding.from_pretrained(embeddings)
            #self.word_embedding.weight.data.copy_(embeddings)

        self.lstm = nn.LSTM(hparams["embedding_dim"], hparams["hidden_dim"], 
                            bidirectional=hparams["bidirectional"],
                            num_layers=hparams["num_layers"], 
                            dropout = hparams["dropout"] if hparams["num_layers"] > 1 else 0)
        
        lstm_output_dim = hparams["hidden_dim"] if hparams["bidirectional"] is False else hparams["hidden_dim"] * 2

        # During training, randomly zeroes some of the elements of the 
        # input tensor with probability hparams.dropout. 
        # This has proven to be an effective technique for regularization and 
        # preventing the co-adaptation of neurons
        self.dropout = nn.Dropout(hparams["dropout"])
        self.classifier = nn.Linear(lstm_output_dim, hparams["num_classes"])

    
    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output
