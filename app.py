import gradio as gr
from clients import send_request
import json
import ast

def save_non_abuse_class(top_classes):
    top_class_json = top_classes['message']
    dic_predict = ast.literal_eval(top_class_json)
    return dic_predict
        
        
    
def prediction(img):
    top_classes = send_request(img_input=img, url='http://127.0.0.1:8000/')
    save_class = save_non_abuse_class(top_classes)
    prediction_str = "\n".join([f"{index+1}. {property} : {round(value*100, 2)}%" for index, (property, value) in enumerate(save_class.items())])
    return prediction_str


with gr.Blocks(css="footer{display:none !important}") as demo:
    with gr.Row():
        prediction_output = gr.Textbox(placeholder="result", label="Prediction")
        gr.Interface(prediction, inputs="image", outputs=prediction_output)




if __name__ == "__main__":
    demo.launch()