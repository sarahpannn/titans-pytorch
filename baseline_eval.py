from transformers import AutoModelForCausalLM

from train_titan_llama import evaluate_model, SlimPajamaDataset

def main():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B").to("cuda")

    eval_dataset = SlimPajamaDataset(
        dataset_name="rokset3/slim_pajama_chunk1",
        tokenizer_name="meta-llama/Meta-Llama-3.1-8B",
        max_length=2048,
        streaming=True,
        split="validation" if "validation" in ["train"] else "train",  # Use train split for now
        num_proc=1
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    average_loss = evaluate_model(model, eval_dataloader, "cuda")
