
from flask import Flask, request, jsonify
from model.model import rag_chain



app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def qa():
    question = request.json["question"]
    result = rag_chain.invoke(question)  # Convert the list to a tuple
    return jsonify(
        {
            "result": result      
        }
    )

if __name__ == "__main__":
    app.run(debug=True)