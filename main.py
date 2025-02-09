from taipy.gui import Gui, State, notify
import os

from src.gutenberg_chatbot.model import generate_text, load_model, RNNModel, device


# Variable context for taipy 'page'
client = None
context = "The following is a conversation with local text-generation model.\n\nHuman: Hello, who are you?\nAI: I am an AI. How can I help you today? "
conversation = {
    "Conversation": [
        "Who are you?",
        "I am a locally trained model. How can I help you today?",
    ]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]


################################
# Chat State + Helpers
################################
def on_init(state: State) -> None:
    """
    Initialize the app.
    """
    # TODO: think about replacing the Ai response with a reference to the trained corpus,
    # or with an init'd response from the trained model to kick things off with flavor
    state.context = (
        "The following is a conversation with a local text-generation model.\n\n"
        "Human: Hello, who are you?\n"
        "AI: I am a locally trained model. How can I help you today?"
    )
    state.conversation = {
        "Conversation": [
            "Who are you?",
            "I am a locally trained model. How can I help you today?",
        ]
    }
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]


################################
# TODO: STUB: Replace with actual local model import/logic
################################

MODEL_DIR = "models"
checkpoint_path = os.path.join(MODEL_DIR, "rnn_model.pth")
model, _ = load_model(checkpoint_path, RNNModel, device)


def generate_response(state: State, prompt: str) -> str:
    """
    Calls local model to generate text.
    """
    if model is None:
        return "I'm currently unavailable for responses. Please try again later."

    generated_text = generate_text(
        model, prompt, generation_length=200, temperature=0.8
    )

    return generated_text


# def update_context(state: State) -> str:
#     """
#     Update the context with the user's message and the model's response.
#     """
#     # add user’s message to the context
#     state.context += f"Human: \n{state.current_user_message}\n\nAI:"

#     # model inference
#     answer = generate_response(state, state.context)

#     # append the answer to the context
#     state.context += answer
#     # keep track of new row index in the conversation table
#     state.selected_row = [len(state.conversation["Conversation"]) + 1]

#     return answer

MAX_CONTEXT_LENGTH = 1000  # Adjust based on your needs


def update_context(state: State) -> str:
    """
    Update the context with the user's message and the model's response.
    """
    # Append only the raw user message (not "Human:")
    new_input = f"{state.current_user_message}\n"

    # Truncate the context while keeping the most recent portion
    truncated_context = (state.context + new_input)[-MAX_CONTEXT_LENGTH:]

    # Model inference using only the meaningful context
    answer = generate_response(state, truncated_context)

    # Append AI response WITHOUT "AI:"
    state.context = truncated_context + answer + "\n"

    # Track conversation history correctly
    state.selected_row = [len(state.conversation["Conversation"]) + 1]

    return answer


def send_message(state: State) -> None:
    """
    Send user's message to model and update context.
    """
    notify(state, "info", "Sending message...")
    answer = update_context(state)
    conv = state.conversation.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Response received!")


def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on message author.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "ai_message"


def reset_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.
    """
    state.past_conversations = state.past_conversations + [
        [len(state.past_conversations), state.conversation]
    ]
    state.conversation = {
        "Conversation": [
            "Who are you?",
            "I am a locally trained model. How can I help you today?",
        ]
    }
    # reset context
    state.context = (
        "The following is a conversation with a local text-generation model.\n\n"
        "Human: Hello, who are you?\n"
        "AI: I am a locally trained model. How can I help you today?"
    )


def tree_adapter(item: list) -> tuple[str, str]:
    """
    Converts element of past_conversations to id and displayed string.
    """
    identifier = item[0]
    conv_data = item[1]["Conversation"]
    if len(conv_data) > 1:
        return (identifier, conv_data[0][:50] + "...")
    return (item[0], "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
    Select a conversation from past_conversations.
    """
    # retrieve that conversation
    saved_conv = state.past_conversations[value[0][0]][1]
    state.conversation = saved_conv

    # rebuild context from the conversation
    state.context = (
        "The following is a conversation with a local text-generation model.\n\n"
        "Human: Hello, who are you?\n"
        "AI: I am a locally trained model. How can I help you today?"
    )

    # if user + AI pairs exist, reconstruct them in the context
    for i in range(2, len(saved_conv["Conversation"]), 2):
        user_msg = saved_conv["Conversation"][i]
        ai_msg = saved_conv["Conversation"][i + 1]
        state.context += f"\nHuman:\n{user_msg}\n\nAI:{ai_msg}"

    # move selected row to the bottom of conversation
    state.selected_row = [len(saved_conv["Conversation"]) + 1]


page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Taipy **Chat**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|row_class_name=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    # model loading here?
    # set it on state or a global var?

    Gui(page).run(
        debug=True, dark_mode=True, use_reloader=True, title="💬 Taipy Chat", port=5001
    )
