from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertModel

class BertBaseModel(nn.Module):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-cased")
        # self.dropout = nn.Dropout(drop_rate)
        # self.linear = nn.Linear(768,NUM_LABELS)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder.forward(input_ids=input_ids,attention_mask=attention_mask)
        enc_hs = outputs[0][-1][:,0,:]  # The last hidden-state is the first element of the output tuple
        # enc_hs = outputs[1]
        enc_hs = enc_hs.reshape(-1,768) # append the embeddings due to the sentence, subject entity context and object entity context
        # rel = self.softmax(self.linear(enc_hs))

        # return rel, enc_hs
        return enc_hs

