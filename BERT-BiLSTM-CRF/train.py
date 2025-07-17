import os
import json
import datetime
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import Adam
from seqeval.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

from dataset import NERDataset, collate_fn
from model import BertBiLstmCrf


def create_experiment_directory():
    """Create a unique directory for this experiment."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = "checkpoints"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def build_label_mapping(train_file, valid_file):
    """Build label to ID mapping from training and validation files."""
    label_set = set()
    for file in [train_file, valid_file]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    label_set.add(line.split()[-1])

    label_set.add("O")
    label_list = ["O"] + sorted([lbl for lbl in label_set if lbl != "O"])
    label_to_id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id_to_label = {idx: lbl for lbl, idx in label_to_id.items()}
    num_labels = len(label_to_id)

    return label_to_id, id_to_label, num_labels


def save_experiment_config(run_dir, timestamp, config_params):
    """Save experiment configuration to JSON file."""
    config = {
        "timestamp": timestamp,
        "batch_size": config_params["batch_size"],
        "epochs": config_params["epochs"],
        "learning_rate": config_params["lr"],
        "weight_decay": config_params["weight_decay"],
        "bert_model": "bert-base-chinese",
        "dropout": 0.1,
        "warmup_proportion": config_params["warmup_proportion"],
        "max_seq_length": config_params["max_seq_length"]
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def evaluate(model, data_loader, id_to_label, device):
    """Evaluate model on given dataset and return F1 score with detailed report."""
    was_training = model.training
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, labels, crf_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            crf_mask = crf_mask.to(device)

            emissions = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch_pred_ids = model.decode(emissions, crf_mask)
            seq_lengths = attention_mask.sum(dim=1).tolist()

            for i, pred_ids in enumerate(batch_pred_ids):
                seq_len = seq_lengths[i]
                true_ids = labels[i, 1:seq_len - 1].tolist()  # Remove CLS/SEP
                true_labels = [id_to_label[idx] for idx in true_ids]
                pred_labels = [id_to_label[idx] for idx in pred_ids]

                all_true.append(true_labels)
                all_pred.append(pred_labels)

    if was_training:
        model.train()

    f1 = f1_score(all_true, all_pred)
    report = classification_report(all_true, all_pred, digits=4)
    return f1, report


def plot_loss_curve(loss_curve, run_dir):
    """Plot and save training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, label="Training Loss", linewidth=1.5)
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    print(f"Loss curve saved to {os.path.join(run_dir, 'loss_curve.png')}")


def save_results(run_dir, val_f1, val_report, test_f1, test_report, total_time):
    """Save training results to text file."""
    results_txt = (
        "Validation Results:\n"
        f"{val_report}\n"
        f"Validation F1: {val_f1:.4f}\n\n"
        "Test Results:\n"
        f"{test_report}\n"
        f"Test F1: {test_f1:.4f}\n\n"
        f"Total Training Time: {total_time / 60:.2f} minutes ({total_time:.1f} seconds)\n"
    )

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(results_txt)
    print(f"Results saved to {os.path.join(run_dir, 'results.txt')}")


def main():
    start_time = time.perf_counter()

    # File paths and configuration
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    test_file = "data/test.txt"
    pretrained_dir = "pretrained_models/bert-base-chinese"

    # Hyperparameters
    config_params = {
        "max_seq_length": 384,
        "batch_size": 16,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_proportion": 0.10,
        "epochs": 15
    }

    # Setup experiment directory
    run_dir, timestamp = create_experiment_directory()

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_dir)

    # Build label mappings
    label_to_id, id_to_label, num_labels = build_label_mapping(train_file, valid_file)

    # Create datasets and dataloaders
    train_dataset = NERDataset(train_file, tokenizer, label_to_id, config_params["max_seq_length"])
    valid_dataset = NERDataset(valid_file, tokenizer, label_to_id, config_params["max_seq_length"])
    test_dataset = NERDataset(test_file, tokenizer, label_to_id, config_params["max_seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=config_params["batch_size"], shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config_params["batch_size"], shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config_params["batch_size"], shuffle=False, collate_fn=collate_fn)

    # Initialize model and optimizer
    model = BertBiLstmCrf(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config_params["lr"], weight_decay=config_params["weight_decay"])

    # Setup learning rate scheduler
    total_steps = len(train_loader) * config_params["epochs"]
    warmup_steps = int(config_params["warmup_proportion"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Save experiment configuration
    save_experiment_config(run_dir, timestamp, config_params)

    # Training loop
    loss_curve = []
    best_val_f1 = 0.0
    best_model_path = "checkpoints/best_model.pt"

    for epoch in range(config_params["epochs"]):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels, crf_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            crf_mask = crf_mask.to(device)

            optimizer.zero_grad()
            emissions = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = model.compute_loss(emissions, labels, crf_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            loss_curve.append(batch_loss)

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{config_params['epochs']} - Step {step + 1}/{len(train_loader)} - Loss: {batch_loss:.4f}")

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_f1, _ = evaluate(model, valid_loader, id_to_label, device)

        epoch_time = time.perf_counter() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - Train Loss: {avg_loss:.4f} - Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (F1: {val_f1:.4f})")

    # Plot loss curve
    plot_loss_curve(loss_curve, run_dir)

    # Final evaluation with best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_val_f1, best_val_report = evaluate(model, valid_loader, id_to_label, device)
    test_f1, test_report = evaluate(model, test_loader, id_to_label, device)

    # Print and save results
    print("\nValidation Results:")
    print(best_val_report)
    print(f"Validation F1: {best_val_f1:.4f}")

    print("\nTest Results:")
    print(test_report)
    print(f"Test F1: {test_f1:.4f}")

    total_time = time.perf_counter() - start_time
    save_results(run_dir, best_val_f1, best_val_report, test_f1, test_report, total_time)


if __name__ == "__main__":
    main()