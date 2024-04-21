from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

app = Flask('carz24')

def generate_plot(y_actual, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_actual, y_pred, alpha=0.5)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
    if not os.path.exists('static'):
        os.makedirs('static')
    plot_filename = f'static/prediction_plot_{uuid.uuid4()}.png'
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            model_type = request.form['model_type']
            if model_type == "Linear":
                from Linear_reg import load_and_preprocess_data, train_model, predict, calculate_metrics
            elif model_type == "RandomForest":
                from random_forest import load_and_preprocess_data, train_model, predict, calculate_metrics
            
            X, y = load_and_preprocess_data()
            model = train_model(X, y)
            y_pred = predict(model, X)
            mse, mae, r2 = calculate_metrics(y, y_pred)

            # Generate plot
            plot_path = generate_plot(y, y_pred)

            return render_template("results.html", mse=mse, mae=mae, r2=r2, plot_path=plot_path)
        except Exception as e:
            return render_template("error.html", error_message=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
