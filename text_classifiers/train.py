from dataset_loader import Dataset
from neural_network import SummarizationModel

dataset = Dataset()
print("Dataset loaded!")
model = SummarizationModel(use_bidir=True, use_attn=True)
model.create(dataset)
print("Model created!")
model.train(dataset)

def seq2summary(input_seq):
    summary=''
    for i in input_seq:
        if i != 0 and i != dataset.label_tokenizer.word_index['start'] and i != dataset.label_tokenizer.word_index['end']:
            summary += dataset.label_tokenizer.index_word[i] + ' '
    return summary

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if i != 0:
            newString += dataset.datapoint_tokenizer.index_word[i]+' '
    return newString

for i, datapoint in enumerate(dataset.datapoint_test):
  print("Review:",seq2text(datapoint))
  print("Original summary:",seq2summary(dataset.label_test[i]))
  print("Predicted summary:",model.create_review(dataset,
                                                 datapoint.reshape(1,dataset.max_len_datapoint)))
  print("\n")
  if i == 10:
      break
