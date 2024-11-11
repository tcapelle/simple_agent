import openai

import weave

weave.init("ejemplo_daniel")

client = openai.OpenAI()


response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=150,
    stream=True  # Enable streaming
)

# Print the streaming response
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()  # Add a newline at the end

