import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "Jawaher/LIAR-fake-news-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Offizielle LIAR-Klassen
id2label = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

def predict_statement(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)[0]

    pred_class_id = torch.argmax(probs).item()
    pred_class_label = id2label[pred_class_id]

    probs_dict = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    return {
        "text": text,
        "predicted_class": pred_class_label,
        "probabilities": probs_dict
    }

result = predict_statement("The economy is doing great.")
print(result)
