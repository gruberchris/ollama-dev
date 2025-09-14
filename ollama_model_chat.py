import ollama

client = ollama.Client()

model = "gruber-coder-1M:latest"
prompt = "What have been the top 10 most popular programming languages since March 2020?"

print(f"Using model: {model}")
print(f"Prompt: {prompt}")

try:
    response = client.generate(model, prompt, stream=True)
except Exception as e:
    print(f"Error generating response: {e}")
    exit(1)

print("\n" + "="*50)

for chunk in response:
    # Handle different types of chunk.response
    if isinstance(chunk.response, str):
        print(chunk.response, end="", flush=True)
    elif isinstance(chunk.response, bytes):
        print(chunk.response.decode('utf-8'), end="", flush=True)
    elif chunk.response is None:
        continue  # Skip empty chunks
    else:
        # Fallback for unexpected types
        print(str(chunk.response), end="", flush=True)

    if chunk.done:
        total_duration_ns = chunk.total_duration
        total_duration_s = total_duration_ns / 1_000_000_000
        print(f"\n\nResponse time: {total_duration_s} seconds")
        print(f"Total prompt tokens: {chunk.prompt_eval_count}")
        print(f"Total response tokens: {chunk.eval_count}")
        print("\n" + "="*50)
        break
