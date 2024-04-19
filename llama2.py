import boto3
import json

prompt_data = """Act as a Bio-Scientist and write a explanation about DNA of a human being."""

# So, with the help of Boto 3, we are connecting our code with AWS.
bedrock = boto3.client(service_name='bedrock-runtime')

# You have to put all the API request (payload) in the format mentioned in the "test.json" file.
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)

# Based on this model ID, we invoke the results from Generative Models of AWS.
# You have to maintain this in AWS account.
model_id = "meta.llama2-70b-chat-v1"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read())

response_text = response_body['generation']

print(response_text)
