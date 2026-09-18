import gradio as gr
import joblib
import pandas as pd
import os

model = joblib.load("Electric_Bill_AC_Fan_Other_model.pkl")


def predict_result(ac, fan, other):

    input_data = pd.DataFrame({
        "AC_Units": [ac],
        "Fan_Units": [fan],
        "Other_Units": [other]
    })

    prediction = model.predict(input_data)[0]

    return f"Predicted Electricity Bill: {prediction}"


demo = gr.Interface(
    fn=predict_result,

    inputs=[
        gr.Number(
            label="Enter the AC Units",
            minimum=0,
            value=1
        ),

        gr.Number(
            label="Enter the Fan Units",
            minimum=0,
            value=1
        ),

        gr.Number(
            label="Enter the Other Units",
            minimum=0,
            value=1
        )
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="Electricity Bill Prediction based on AC, Fan and Other Units",

    description="Prediction of electricity bill based on AC, Fan and Other Units."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
