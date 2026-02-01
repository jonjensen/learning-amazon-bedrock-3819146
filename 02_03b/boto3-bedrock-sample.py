#Imports
import boto3
from botocore.exceptions import ClientError

#Create the bedrock client
client = boto3.client('bedrock-runtime')

#Setting the prompt
user_message = """Command: Write me a blog post about coaching employees as a leader.

Blog:
"""

#Model specification
model_id = "amazon.nova-lite-v1:0"
accept = "application/json"
contentType = accept

conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
