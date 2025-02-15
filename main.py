from taipy.gui import Gui, State, notify

from gutenberg_chatbot.model import generate_text
from gutenberg_chatbot.model_bootstrap import load_default_model_and_vocab

# Necessary variable context for taipy 'page' object upon startup
context = ""
conversation = {}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]

# default model loading
model, c2ix, ix2c = load_default_model_and_vocab()


################################
# Chat State + Helpers
################################
def on_init(state: State) -> None:
    """
    Initialize the app.
    """
    state.context = ""
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


def generate_response(state: State, prompt: str) -> str:
    """
    Calls local model to generate text.
    """
    if model is None:
        return "I'm currently unavailable for responses. Please try again later."

    generated_text = generate_text(
        model, prompt, c2ix, ix2c, generation_length=200, temperature=0.8
    )

    return generated_text


def build_prompt_from_turns(state: State, num_turns: int = 3) -> str:
    """
    Construct a prompt from the last `num_turns` user-AI pairs in the conversation.
    """
    conversation_list = state.conversation["Conversation"]
    # Each turn is 2 messages, so the last N turns is the last 2*N messages.
    start_idx = max(0, len(conversation_list) - 2 * num_turns)

    # consider a system prompt
    # system_prompt = "You are a helpful, locally trained AI. Please answer politely.\n\n"
    # prompt = system_prompt

    # construct a text block.
    prompt = ""
    for i in range(start_idx, len(conversation_list), 2):
        user_msg = conversation_list[i]
        ai_msg = ""
        if i + 1 < len(conversation_list):
            ai_msg = conversation_list[i + 1]
        prompt += (
            f"{user_msg}\n{ai_msg}\n"  # retain `Human: {user_msg} AI: {ai_msg} format?
        )

    return prompt


def update_context(state: State) -> str:
    # build prompt from last N turns
    prompt = build_prompt_from_turns(state, num_turns=3)
    # adding the new user message to the prompt
    prompt += state.current_user_message

    # generate answer from the local model
    answer = generate_response(state, prompt)

    # store the entire prompt in `state.context` even though it's not needed
    state.context = prompt + answer

    # let the table know about it
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer


def send_message(state: State) -> None:
    """
    Send user's message to model and update context.
    User types a message into the input field ({current_user_message} in the UI).
    Taipy calls send_message(state) when the user submits
    """
    notify(state, "info", "Sending message...")
    answer = update_context(state)  # generate a response from the model
    conv = state.conversation.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""  # clear the old user message
    # table bound to state.conversation now has new row of user+model exchange
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
        # retain `Human: {user_msg} AI: {ai_msg} format?
        state.context += f"{user_msg}\n{ai_msg}\n"

    # move selected row to the bottom of conversation
    state.selected_row = [len(saved_conv["Conversation"]) + 1]


page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# RNN **Chat**{: .color-primary} # {: .logo-text}
This is my chat interface using a locally trained RNN.
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>

<|Train Model|button|class_name=fullwidth plain|on_action=trigger_training|>

### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|row_class_name=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=input-container|
<|{current_user_message}|input|label=Write here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page, css_file="static/main.css").run(
        debug=True, dark_mode=True, use_reloader=True, title="💬 RNN Chat", port=5001
    )
