import boto3
import json

# Claude is for content generation

prompt_data = """Act as a Bio-Scientist and write a explanation about DNA of a human being."""

# So, with the help of Boto 3, we are connecting our code with AWS.
bedrock = boto3.client(service_name='bedrock-runtime')

payload = {
    "prompt": prompt_data,
    "maxTokens": 512,
    "temperature": 0.8,
    "topP": 0.8
}

body = json.dumps(payload)

model_id = "ai21.j2-mid-v1"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept='application/json',
    contentType='application/json'
)

response_body = json.loads(response.get("body").read())
response_text = response_body.get('completions')[0].get("data").get("text")
print(response_text)
