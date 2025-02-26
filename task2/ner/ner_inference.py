import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import nltk
import inflect

nltk.download('punkt', quiet=True)
inflect_engine = inflect.engine()

# Loading the models and tokenizer from Hugging Face
model_id = "sofibrezden/animal-ner"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)


def to_singular(entity):
    """
    Converts each word in the entity from plural to singular.
     "dogs" to "dog",
    """
    words = entity.split()
    singular_words = []
    for word in words:
        singular = inflect_engine.singular_noun(word)
        singular_words.append(singular if singular else word)
    return " ".join(singular_words)


def process_sentence(sentence):
    """
        Processes a sentence:
        - Tokenizes the sentence with offset mapping.
        - Passes tokens through the models to get predicted labels.
        - Groups consecutive tokens that belong to the "ANIMAL" entity.
        - Converts the obtained entities from plural to singular.
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=2)[0].tolist()
    predicted_labels = [model.config.id2label[pred_id] for pred_id in predicted_ids]

    entities = []
    current_entity = ""
    for token, label, offset in zip(tokens, predicted_labels, offset_mapping):
        if offset[0] == offset[1] == 0:
            continue
        # If the label contains "ANIMAL", the token is part of the entity
        if "ANIMAL" in label:
            # If the token is a subword of the entity, add it to the accumulated entity text
            if token.startswith("##"):
                current_entity += token[2:]
            else:
                if current_entity:
                    current_entity += " " + token
                else:
                    current_entity = token
        else:
            # If the token is not part of an entity and there is an accumulated entity, save it
            if current_entity:
                entities.append(current_entity)
                current_entity = ""
    if current_entity:
        entities.append(current_entity)

    # Convert each entity from plural to singular
    return [to_singular(entity) for entity in entities]


def extract_animals(text):
    sentences = nltk.sent_tokenize(text)
    animals = []
    for sent in sentences:
        detected = process_sentence(sent)
        corrected_animals = []
        for animal in detected:
            singular_animal = to_singular(animal)
            if singular_animal not in corrected_animals:
                corrected_animals.append(singular_animal)
        animals.extend(corrected_animals)
    return animals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect animal names in text.")
    parser.add_argument("text", type=str, help="Input text for animal detection.")
    args = parser.parse_args()

    animals = extract_animals(args.text)
    print("Detected animals:", animals)