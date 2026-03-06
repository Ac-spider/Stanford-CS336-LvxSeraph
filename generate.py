import torch
from model import TransformerLM, softmax
from tokenizer import Tokenizer

def load_model_and_tokenizer(checkpoint_path, vocab_path, merges_path):

    tokenizer = Tokenizer.from_files(vocab_path,merges_path,special_tokens=["<|endoftext|>"])

    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, device=device)

    print(f"正在加载检查点: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path,map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    return model,tokenizer,device,context_length

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, context_length:int,temperature: float = 1.0, top_p: float = 0.9,
                  device: str = "cpu"):

    token_ids = tokenizer.encode(prompt)
    x = torch.tensor([token_ids],dtype=torch.long,device=device)

    print(f"\n[Prompt]: {prompt}")
    print("-" * 40)

    end_token_bytes = b"<|endoftext|>"
    end_id = tokenizer.inverse_vocab[end_token_bytes] if end_token_bytes in tokenizer.inverse_vocab else None

    with (torch.no_grad()):
        for _ in range(max_new_tokens):
            logits = model(x[:, -context_length:])[0, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            else:
                probs = softmax(logits/temperature, -1)
                if top_p < 1:
                    sorted_probs,sorted_indicies = torch.sort(probs,-1,descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs,-1)
                    sorted_probs_to_remove = cumsum_probs > top_p

                    sorted_probs_to_remove[1:] = sorted_probs_to_remove[:-1].clone()
                    sorted_probs_to_remove[0] = False

                    indicies_to_remove = torch.zeros_like(probs, dtype=torch.bool
                                                          ).scatter_(-1,sorted_indicies,sorted_probs_to_remove)
                    probs[indicies_to_remove] = 0

                    probs = probs / (torch.sum(probs,-1,keepdim=True))

                next_token = torch.multinomial(probs,1).unsqueeze(0)
            x = torch.cat((x,next_token),-1)

            if next_token.item() == end_id:
                break

        generated = tokenizer.decode(x[0].tolist())

    return generated



if __name__ == '__main__':
    vocab_file = "outputs/TinyStories_vocab.pkl"
    merges_file = "outputs/TinyStories_merges.pkl"
    import os


    for step in range(1000, 10001, 1500):
        checkpoint_file = f"checkpoints/model_step_{step}.pt"

        if not os.path.exists(checkpoint_file):
            print(f"找不到检查点 {checkpoint_file}。请先运行训练或修改路径。")
            continue

        model, tokenizer, device,context_length = load_model_and_tokenizer(checkpoint_file, vocab_file, merges_file)

        prompt_text = "Once upon a time, there was an evil dragon who"

        generated_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=100,
            context_length=context_length,
            temperature=0.8,
            top_p=0.9,
            device=device
        )

        print(f'{step}/10000,\n{generated_output}')























