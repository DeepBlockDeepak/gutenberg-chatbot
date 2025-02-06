from taipy.gui import Gui, State


################################
# TODO: STUB: Replace with actual local model import/logic
################################
def generate_response(state: State, prompt: str) -> str:
    """
    <STUB>
    Eventually, will call local model to generate text.
    """
    # TODO: Replace w/t model inference code later.
    return "Hello! I'm a placeholder response from the local model."


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


def update_context(state: State) -> str:
    """
    Update the context with the user's message and the model's response.
    """
    # add user’s message to the context
    state.context += f"Human: \n{state.current_user_message}\n\nAI:"

    # model inference
    answer = generate_response(state, state.context)

    # append the answer to the context
    state.context += answer
    # keep track of new row index in the conversation table
    state.selected_row = [len(state.conversation["Conversation"]) + 1]

    return answer


page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Taipy **Chat**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":

    Gui(page).run(
        debug=True, dark_mode=True, use_reloader=True, title="💬 Taipy Chat", port=5001
    )
