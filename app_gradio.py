import gradio as gr
import joblib
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures

model = joblib.load("Electric_Bill_AC_Fan_Other_model.pkl")

poly = PolynomialFeatures(degree=2)

poly.fit([[0, 0, 0]])


def predict_result(ac, fan, other):

    input_data = pd.DataFrame({
        "AC_Units": [ac],
        "Fan_Units": [fan],
        "Other_Units": [other]
    })

    input_poly = poly.transform(input_data)

    prediction = model.predict(input_poly)[0]

    return f"Predicted Electricity Bill: ₹{prediction:.2f}"


demo = gr.Interface(
    fn=predict_result,

    inputs=[
        gr.Number(label="Enter the AC Units", minimum=0, value=1),
        gr.Number(label="Enter the Fan Units", minimum=0, value=1),
        gr.Number(label="Enter the Other Units", minimum=0, value=1)
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="Electricity Bill Prediction",
    description="Predict electricity bill based on AC, Fan and Other Units."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
