# Reuters Newswires Classifier

A multi-class classification application that categorizes news articles into **46 mutually exclusive topics** (e.g., 'earn', 'acq', 'crude', 'grain'). This project features a **FastAPI** backend with a custom **NumPy-based inference engine**, demonstrating deep learning concepts from scratch without relying on heavy frameworks like TensorFlow or PyTorch for production.

## 📌 Project Overview

The goal of this project is to automatically label Reuters newswires with their corresponding topics. The system uses a neural network trained on the Reuters dataset.

**Key Highlights:**
* **Custom Inference Engine:** The entire forward pass (matrix multiplications, ReLU, Softmax) is implemented manually using **NumPy**, making the production code lightweight and transparent.
* **46-Class Classification:** Capable of distinguishing between a wide variety of news topics.
* **Full-Stack Application:** Includes a FastAPI backend and a clean HTML/JavaScript frontend for real-time user interaction.

## 📂 Repository Structure

- `main.py`: The core FastAPI application. It loads the model weights, performs text vectorization, runs the custom forward pass, and serves the API endpoints.
- `static/index.html`: A responsive frontend where users can paste news text and view the predicted category.
- `models/model_assets.npz`: Compressed NumPy archive containing the trained weights (`W1`, `b1`, etc.) and the word index dictionary required for preprocessing.
- `reuters_nn_model.ipynb`: The Jupyter Notebook used for training the model. It covers data loading, preprocessing, and the training loop implementation.

## 🛠️ Technologies Used

- **Backend:** Python, FastAPI, Uvicorn
- **Computation:** NumPy (used for all matrix operations and activation functions during inference)
- **Frontend:** HTML5, CSS, JavaScript (Fetch API)
- **Data:** Reuters Dataset (via Keras datasets for training)

## 🧠 Model Architecture

The underlying model is a **Multilayer Perceptron (MLP)** with the following structure:

1.  **Input Layer:** Bag-of-Words representation (Vectorized text, max 10,000 words).
2.  **Hidden Layers:** Two dense layers using **ReLU** activation.
3.  **Output Layer:** A dense layer with 46 units using **Softmax** activation to output probabilities for each category.

*Note: While the model structure is standard, the specific implementation in `main.py` manually calculates `Z = W.dot(X) + b` and applies activations without high-level Keras/Torch layers.*

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Install Dependencies**
    You will need FastAPI, Uvicorn, and NumPy.
    ```bash
    pip install fastapi uvicorn numpy pydantic
    ```

3.  **Start the Server**
    ```bash
    python main.py
    ```
    *Note: Ensure `main.py` contains the block `uvicorn.run(...)`, otherwise run with `uvicorn main:app --reload`.*

4.  **Access the App**
    Open your browser and navigate to:
    `http://localhost:8000` (or the port specified in your console).

## 📊 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "The company reported a profit of 5 million dollars..."
}
