import streamlit as st
from openai import OpenAI 

client = OpenAI(api_key="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")

def chatbot_ui():
    st.subheader("AI Stock Assistant")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show old messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about stocks...")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Get response
        response = chatbot_response(user_input)

        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)


# -------------------------------
# Chatbot Logic (SAFE VERSION)
# -------------------------------
def chatbot_response(query):
    query = query.lower()
    last = st.session_state.get("last_prediction", None)

    # -------------------------------
    # 1️⃣ Smart Understanding (keywords + intent)
    # -------------------------------
    if any(word in query for word in ["prediction", "predict", "forecast"]):
        return (
            "📊 Prediction means forecasting future stock prices.\n\n"
            "This app uses a Random Forest Machine Learning model "
            "to predict whether price will go UP 📈 or DOWN 📉 "
            "based on historical data."
        )

    elif any(word in query for word in ["why", "reason", "explain"]) and last:
        data = last["input"]
        result = last["result"]

        return (
            f"🤖 Here's why:\n\n"
            f"- Open: {data['Open']}\n"
            f"- Close: {data['Close']}\n"
            f"- Volume: {data['Volume']}\n\n"
            f"📊 Prediction: {result}\n\n"
            f"The model detected patterns in price movement and volume trends."
        )

    elif any(word in query for word in ["machine learning", "ml"]):
        return "🤖 Machine Learning allows computers to learn patterns from data and make predictions."

    elif any(word in query for word in ["random forest", "model"]):
        return "🌳 Random Forest is an ensemble algorithm using multiple decision trees to improve prediction accuracy."

    elif any(word in query for word in ["feature", "input"]):
        return "📊 The model uses Open, High, Low, Close, and Volume as input features."

    elif any(word in query for word in ["buy", "invest"]):
        return "📈 Buy when trend is upward and volume is increasing."

    elif any(word in query for word in ["sell"]):
        return "📉 Sell when trend weakens or reverses."

    elif any(word in query for word in ["risk", "safe"]):
        return "⚠️ Always diversify investments and manage risk carefully."

    elif any(word in query for word in ["project", "about"]):
        return (
            "💡 This is a Stock Market Prediction project using Machine Learning.\n"
            "It predicts stock movement and includes visualization, data storage, and an AI chatbot."
        )

    elif any(word in query for word in ["hello", "hi", "hey"]):
        return "👋 Hello! Ask me anything about stocks or this project."

    # -------------------------------
    # 2️⃣ FALLBACK (Human-like response)
    # -------------------------------
    else:
        return (
            "I understand your question, but I specialize in:\n\n"
            "• Stock prediction \n"
            "• Machine learning \n"
            "• Investment strategies \n\n"
            " Try asking something related to these!"
        )