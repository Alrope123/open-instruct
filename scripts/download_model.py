from transformers import AutoModel, AutoTokenizer

def download_and_save_model(model_name, save_directory):
    # Load the model and tokenizer using the model name
    model = AutoModel.from_pretrained(model_name, cache_dir=save_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)

    # Save the model and tokenizer to the specified directory
    # model.save_pretrained(save_directory)
    # tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer have been saved to {save_directory}")

# Specify the model and the directory where you want to save it
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Example model
save_directory = "tmp/model"  # Directory to save the model

download_and_save_model(model_name, save_directory)
