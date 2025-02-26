import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from seqeval.metrics import precision_score, recall_score, f1_score
from datasets import Dataset as HFDataset

# Define labels and their indices
ENTITY_LABELS = ['ANIMAL', 'O']
LABEL_TO_ID = dict(zip(ENTITY_LABELS, range(len(ENTITY_LABELS))))
ID_TO_LABEL = dict(zip(range(len(ENTITY_LABELS)), ENTITY_LABELS))

# Tokenize and align labels function
def process_and_align_tokens(tokens, labels, tokenizer, label_map, ignore_index=-100):
    tokenized_output = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        padding=True
    )
    tokenized_output.pop("offset_mapping")

    all_labels = []
    for index in range(len(tokenized_output.encodings)):
        word_indices = tokenized_output.word_ids(batch_index=index)
        previous_idx = None
        token_labels = []
        for word_idx in word_indices:
            if word_idx is None:
                token_labels.append(ignore_index)  # Ignore special tokens
            elif word_idx != previous_idx:
                token_labels.append(label_map.get(labels[index][word_idx], ignore_index))
            else:
                token_labels.append(label_map.get(labels[index][word_idx], ignore_index))
            previous_idx = word_idx
        all_labels.append(token_labels)

    tokenized_output["labels"] = all_labels
    return tokenized_output

# to evaluate models predictions
def evaluate_predictions(eval_results):
    logits, actual_labels = eval_results
    predicted_labels = np.argmax(logits, axis=2)

    true_sequences = []
    predicted_sequences = []
    for preds, actual in zip(predicted_labels, actual_labels):
        actual_sequence = []
        pred_sequence = []
        for pred, label in zip(preds, actual):
            if label != -100:
                actual_sequence.append(ID_TO_LABEL[label])
                pred_sequence.append(ID_TO_LABEL[pred])
        true_sequences.append(actual_sequence)
        predicted_sequences.append(pred_sequence)

    precision = precision_score(true_sequences, predicted_sequences)
    recall = recall_score(true_sequences, predicted_sequences)
    f1 = f1_score(true_sequences, predicted_sequences)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    return {"precision": precision, "recall": recall, "f1": f1}

def run_training(args):
    # Load and preprocess dataset
    dataset_df = pd.read_csv(args.dataset_path)
    dataset_df["Sentence"] = dataset_df["Sentence"].str.split()
    dataset_df["Tags"] = dataset_df["Tags"].str.split()

    sentences_list = dataset_df["Sentence"].tolist()
    tags_list = dataset_df["Tags"].tolist()

    # Split dataset into training and validation sets
    train_sentences, val_sentences, train_tags, val_tags = train_test_split(
        sentences_list, tags_list, test_size=args.test_size, random_state=42
    )

    # Tokenize and align labels
    bert_tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    train_tokens = process_and_align_tokens(train_sentences, train_tags, bert_tokenizer, LABEL_TO_ID)
    val_tokens = process_and_align_tokens(val_sentences, val_tags, bert_tokenizer, LABEL_TO_ID)

    # Create Hugging Face datasets
    training_dataset = HFDataset.from_dict(train_tokens)
    validation_dataset = HFDataset.from_dict(val_tokens)

    # Initialize BERT models for token classification
    bert_model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(ENTITY_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )

    # Configure training arguments
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=500,
        save_total_limit=1,
        logging_dir=args.logging_dir,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        disable_tqdm=False,
        push_to_hub=True,
        hub_model_id="sofibrezden/animal-ner",
        hub_strategy="end"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=bert_model,
        args=trainer_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=bert_tokenizer,
        compute_metrics=evaluate_predictions,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # Train the models and push to Hugging Face Hub
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="tagged_dataset.csv", help="Path to the tagged dataset CSV")
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to be used for validation")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")

    args = parser.parse_args()
    run_training(args)
