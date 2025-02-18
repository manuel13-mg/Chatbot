import re
from openai import OpenAI
import streamlit as st
import time

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=st.secrets["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"]  # Store API key in Streamlit secrets
)


def chat_with_llm(prompt, model="deepseek-ai/deepseek-r1", temperature=0.7, top_p=0.7, max_tokens=1024):
    """
    Interacts with the specified LLM and returns a stream of tokens.

    Args:
        prompt (str): The user's input prompt.
        model (str): The name of the LLM model to use.
        temperature (float): Controls randomness of the output.
        top_p (float): Controls nucleus sampling.
        max_tokens (int): Maximum number of tokens.

    Yields:
        str: A single token from the LLM's response.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True  # Enable streaming
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
              yield chunk.choices[0].delta.content

    except Exception as e:
        st.error(f"Error during LLM interaction: {e}")
        yield None


def main():
    """
    Main function to run the general chatbot using Streamlit with dynamic display.
    """
    st.title("General Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What's on your mind?"):
        # Display user message in chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the LLM (streaming)
        response_container = st.empty()
        full_response = ""
        with st.chat_message("assistant"):
          for chunk in chat_with_llm(prompt):
              if chunk is not None: # Check for errors
                  full_response += chunk
                  response_container.markdown(full_response + "▌") # Dynamic display
              else:
                  st.warning("Sorry, I encountered an error. Please try again.")
                  full_response = "Sorry, I encountered an error. Please try again."
                  break  # Exit the loop on error
          response_container.markdown(full_response)  # Final response

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
