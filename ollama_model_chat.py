import datetime

import ollama
from ollama import ProcessResponse


def report_stats(response_chunk):
    total_duration_ns = response_chunk.total_duration
    total_duration_ms = total_duration_ns / 1_000_000
    total_duration_ms_rounded = round(total_duration_ms, 2)
    load_duration_ns = response_chunk.load_duration
    load_duration_ms = load_duration_ns / 1_000_000
    load_duration_ms_rounded = round(load_duration_ms, 2)
    prompt_eval_count = response_chunk.prompt_eval_count
    prompt_eval_duration_ns = response_chunk.prompt_eval_duration
    prompt_eval_duration_ms = prompt_eval_duration_ns / 1_000_000
    prompt_eval_duration_ms_rounded = round(prompt_eval_duration_ms, 2)
    prompt_eval_duration_sec = prompt_eval_duration_ms / 1_000
    prompt_eval_rate = prompt_eval_count / prompt_eval_duration_sec
    prompt_eval_rate_rounded = round(prompt_eval_rate, 2)
    eval_count = response_chunk.eval_count
    eval_duration_ns = response_chunk.eval_duration
    eval_duration_ms = eval_duration_ns / 1_000_000
    eval_duration_ms_rounded = round(eval_duration_ms, 2)
    eval_duration_sec = eval_duration_ms / 1_000
    eval_rate = eval_count / eval_duration_sec
    eval_rate_rounded = round(eval_rate, 2)

    print(f"\n\ntotal duration: {total_duration_ms_rounded}ms")
    print(f"load duration: {load_duration_ms_rounded}ms")
    print(f"prompt eval count: {prompt_eval_count} tokens")
    print(f"prompt eval duration: {prompt_eval_duration_ms_rounded}ms")
    print(f"prompt eval rate: {prompt_eval_rate_rounded} tokens/sec")
    print(f"eval count: {eval_count} tokens")
    print(f"eval duration: {eval_duration_ms_rounded}ms")
    print(f"eval rate: {eval_rate_rounded} tokens/sec")


def humanize_timedelta(td: datetime.timedelta) -> str:
    seconds = int(td.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds from now"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minutes from now"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hours from now"
    else:
        days = seconds // 86400
        return f"{days} days from now"


def process_report(process_response: ProcessResponse):
    for ollama_model in process_response.models:
        print(f"\n\nmodel: {ollama_model.name}")
        print(f"context length: {ollama_model.context_length} bytes")

        current_time = datetime.datetime.now(datetime.timezone.utc)
        time_until_expiration = ollama_model.expires_at - current_time
        print(f"expiration: {humanize_timedelta(time_until_expiration)}")

        print(f"family: {ollama_model.details.family}")
        print(f"parameter size: {ollama_model.details.parameter_size}")
        print(f"quantization: {ollama_model.details.quantization_level}\n")
    
    
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
        print("\n" + "=" * 50)
        report_stats(chunk)
        process_report(client.ps())
        break
