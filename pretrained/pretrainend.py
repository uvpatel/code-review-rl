from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")


# model.save_pretrained("./bert-base-uncased")
# tokenizer.save_pretrained("./bert-base-uncased")


