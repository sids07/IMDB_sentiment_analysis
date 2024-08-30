import gradio as gr
import requests

def get_response(text, host, port):
    
    host = host.replace("http://","")
    url = "http://" + host +":" +port + "/get_sentiment"
    
    response = requests.post(
        url = url,
        json = {
            "text": text
        }
    )
    return response.json()["sentiment"] 

if __name__ == "__main__":
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                
                text = gr.Text(label= "Enter Text to be classified:")
                
                button = gr.Button("Get Sentiment")
                
            with gr.Column():    
                host = gr.Text(label= "Host for inference API", value="0.0.0.0")
                port = gr.Text(label = "Port for inference API", value=9099)
        with gr.Row():
            with gr.Column():        
                sentiment = gr.Text(label="Sentiment", interactive=False)
        
        button.click(get_response, inputs = [text, host, port], outputs =[sentiment])
        
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 9999
    )