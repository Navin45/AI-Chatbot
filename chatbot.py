import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    chat_history_ids = None
    print("start chatting with the bot! Type 'quit' to stop")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print(" Exiting chat.")
                break

            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            if chat_history_ids is not None:
                bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
            else:
                bot_input_ids = new_input_ids

            attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
            chat_history_ids = model.generate(
                bot_input_ids,
                attention_mask=attention_mask,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id
            )

            bot_response = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
            )

            print(f"Bot: {bot_response}")

        except Exception as e:
            print(" Error occurred:", e)

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=1) if chat_history_ids is None else new_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id =tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(f"Bot: {bot_response}")

if __name__ == "__main__":
    main()
