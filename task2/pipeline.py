import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ner.ner_inference import extract_animals
from image_classifier.img_inference import classify_image

ner_model_id = "sofibrezden/animal-ner"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_id)


def main(text, image_path):
    detected_animals = extract_animals(text)
    predicted_animal, _ = classify_image(image_path)
    match = predicted_animal in detected_animals
    print(f"NER detected: {detected_animals}, Image classifier: {predicted_animal}, Match: {match}")
    return match

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text describing the image")
    parser.add_argument("image_path", type=str, help="Path to the image")
    args = parser.parse_args()
    main(args.text, args.image_path)
