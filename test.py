from gradio_client import Client, file

client = Client("http://3.236.139.198:7860/")

# Test the endpoint by ensuring the input parameters match those expected by the Gradio app
result = client.predict(
    file('test1.jpg'),  # for imgs
    file('dress1.jpg'),  # for garm_img
    "dresses",  # for category
    "Maxi Dress",  # for prompt
    api_name="/virtual-fit"
)

print(result)

