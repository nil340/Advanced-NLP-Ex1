from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import torch
from datasets import load_dataset
import click
from evaluate import load
from transformers import DataCollatorWithPadding


def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def init_model(path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased').to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path).to(
            device)

    return model


def get_train_val_test_split(train_size=-1, val_size=-1, test_size=-1):
    mrpc = load_dataset("glue", "mrpc")
    mrpc_train = mrpc['train'] if train_size == -1 else mrpc['train'].select(
        range(train_size))
    mrpc_val = mrpc['validation'] if val_size == -1 else mrpc[
        'validation'].select(range(val_size))
    mrpc_test = mrpc['test'] if test_size == -1 else mrpc['test'].select(
        range(test_size))
    return mrpc_train, mrpc_val, mrpc_test


def tokenize_batch(data):
    return tokenizer(data["sentence1"], data["sentence2"], truncation=True,
                     padding=False)


def set_training_args(num_train_epochs, batch_size, lr):
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1)

    return training_args


def get_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                dc, compute_metrics):
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=dc,
                      compute_metrics=compute_metrics)
    return trainer


def compute_metrics(eval_pred):
    metric = load("glue", "mrpc")
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)


tokenizer = init_tokenizer()


@click.command()
@click.option("--max_train_samples", type=int, default=-1)
@click.option("--max_eval_samples", type=int, default=-1)
@click.option("--max_predict_samples", type=int, default=-1)
@click.option("--num_train_epochs", type=int)
@click.option("--lr", type=float)
@click.option("--batch_size", type=int)
@click.option("--do_train", is_flag=True)
@click.option("--do_predict", is_flag=True)
@click.option("--model_path", type=str, default=None)
def main(max_train_samples, max_eval_samples, max_predict_samples,
         num_train_epochs, lr, batch_size, do_train, do_predict, model_path):

    train_dataset, val_dataset, test_dataset = get_train_val_test_split(
        max_train_samples, max_eval_samples, max_predict_samples)

    cols_to_remove = ["sentence1", "sentence2", "idx"]

    train_dataset = train_dataset.map(tokenize_batch,
                                      batched=True).remove_columns(
        cols_to_remove)
    val_dataset = val_dataset.map(tokenize_batch, batched=True).remove_columns(
        cols_to_remove)
    test_dataset = test_dataset.map(tokenize_batch,
                                    batched=True).remove_columns(
        cols_to_remove)

    if do_train:
        model = init_model()

        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        training_args = set_training_args(num_train_epochs, batch_size, lr)

        trainer = get_trainer(model, training_args, train_dataset, val_dataset,
                              tokenizer, data_collator, compute_metrics)

        train_results = trainer.train()
        if model_path is not None:
            trainer.save_model(model_path)
        else:
            trainer.save_model("./trained_model")


    if do_predict:

        raw_test = load_dataset("glue", "mrpc")["test"]

        model = init_model(model_path)

        trainer = Trainer(model=model, tokenizer=tokenizer,
                          compute_metrics=compute_metrics)

        model.eval()

        testset_results = trainer.predict(test_dataset).predictions.argmax(
            axis=-1)

        with open("predictions.txt", "w", encoding="utf-8") as out:
            for sentence_1, sentence_2, prediction in zip(raw_test["sentence1"],
                                                          raw_test["sentence2"],
                                                          testset_results):
                out.write(f"{sentence_1}###{sentence_2}###{int(prediction)}\n")


if __name__ == '__main__':
    main()