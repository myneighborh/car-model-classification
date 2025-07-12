import os
import gradio as gr
import inference


inference.main()

predict = inference.predict
examples = inference.examples
class_names = inference.class_names

with gr.Blocks() as demo:
    gr.Markdown(
        "# Vehicle Model Classifier\n"
        "Select one of the example images below or upload your own vehicle image,\n"
        "and the AI will predict the **exact model name** from **396 real-world vehicle classes**."
    )

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        label_input = gr.Dropdown(choices=sorted(class_names), label="Actual Label")

    with gr.Row():
        predict_btn = gr.Button("Predict")

    with gr.Row():
        result_output = gr.Textbox(label="Prediction Result")

    predict_btn.click(fn=predict, inputs=[img_input, label_input], outputs=result_output)

    gr.Examples(
        examples=examples,
        inputs=[img_input, label_input]
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=True)
