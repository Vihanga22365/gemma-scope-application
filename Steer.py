# File: steer_chatbot.py

import streamlit as st
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(layout="wide")

# Initialize Session State
if "default_memory" not in st.session_state:
    st.session_state.default_memory = ConversationBufferMemory()
if "steered_memory" not in st.session_state:
    st.session_state.steered_memory = ConversationBufferMemory()
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []  # To store selected descriptions, layer, index, and strength
if "available_descriptions" not in st.session_state:
    st.session_state.available_descriptions = []  # To temporarily store descriptions for a query

# API details
API_URL = "https://www.neuronpedia.org/api/steer-chat"
SEARCH_API_URL = "https://www.neuronpedia.org/api/explanation/search-model"
MODEL_ID = "gemma-2-9b-it"
HEADERS = {"Content-Type": "application/json", "X-Api-Key": "YOUR_TOKEN"}


if st.sidebar.button('Microscope', use_container_width=True):
    switch_page("Microscope")
    
# Streamlit UI
st.title("Steer With SAE Features (Chat)")
st.sidebar.title("Settings")

# User input for search query
st.sidebar.markdown("### Search for Features")
query = st.sidebar.text_input("Enter Feature:", key="query_input", placeholder="Search for features...")

# Search and display results
if st.sidebar.button("Search"):
    if len(query) >= 3:
        try:
            # Call the Search API
            search_payload = {"modelId": MODEL_ID, "query": query}
            search_response = requests.post(SEARCH_API_URL, json=search_payload, headers=HEADERS)
            search_response.raise_for_status()
            search_data = search_response.json()

            explanations = search_data.get("results", [])
            if explanations:
                # Extract and display explanation descriptions for user selection
                st.session_state.available_descriptions = [
                    {
                        "description": exp["description"],
                        "layer": exp["layer"],
                        "index": exp["index"],
                        "strength": exp.get("strength", 40),  # Default strength to 40 if not provided
                    }
                    for exp in explanations
                ]
            else:
                st.sidebar.error("No features found.")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Search API request failed: {e}")
    else:
        st.sidebar.error("Query must be at least 3 characters long.")

# Handle description selection
if st.session_state.available_descriptions:
    descriptions = [desc["description"] for desc in st.session_state.available_descriptions]
    selected_description = st.sidebar.selectbox("Select feature", [""] + descriptions, key="description_select")

    if selected_description:
        # Find the corresponding feature and add it to the selected features
        
        unique_session_key = f"remove_session_{selected_description}"
        st.session_state[unique_session_key] = selected_description
        
        feature = next(
            (desc for desc in st.session_state.available_descriptions if desc["description"] == selected_description),
            None,
        )
        if feature and feature not in st.session_state.selected_features:
            st.session_state.selected_features.append(feature)
            st.session_state.available_descriptions = []  # Clear temporary storage after selection
            del st.session_state["description_select"]  # Remove dropdown from UI
            st.sidebar.success(f"Feature added: {selected_description}")


# Display selected descriptions with sliders and remove buttons
updated_features = []
if st.session_state.selected_features:
    for feature in st.session_state.selected_features:
        remove_clicked = False

        col1, col2 = st.sidebar.columns([4, 1])  # Create two columns for slider and button
        
        unique_remove_session_key = f"remove_session_{feature['description']}"
               

        with col2:
            if f"remove_session_{feature['description']}" in st.session_state:
                remove_button = st.button("❌", key=f"remove_{feature['description']}")
                if remove_button:
                    remove_clicked = True

        if not remove_clicked:
            with col1:
                # Display slider only if the feature is not marked for removal
                # if feature not in updated_features:
                    feature["strength"] = st.slider(
                        f"Strength for '{feature['description']}'",
                        min_value=-100,
                        max_value=100,
                        value=feature["strength"],
                        key=f"strength_{feature['description']}",
                    )
            updated_features.append(feature)
        else:
            # Remove slider session state for the feature
            if f"strength_{feature['description']}" in st.session_state:
                del st.session_state[f"strength_{feature['description']}"]
            if f"remove_session_{feature['description']}" in st.session_state:
                del st.session_state[f"remove_session_{feature['description']}"]
        

    # Update session state with the remaining features
    st.session_state.selected_features = updated_features
else:
    st.sidebar.markdown("No features selected yet.")
    


# User input for other settings
temperature = st.sidebar.slider("Temperature", -2.0, 2.0, 0.0)
n_tokens = st.sidebar.number_input("Tokens", value=48, step=1)
freq_penalty = st.sidebar.number_input("Frequency Penalty", value=2, step=1)
seed = st.sidebar.number_input("Seed", value=16, step=1)
strength_multiplier = st.sidebar.number_input("Strength Multiplier", value=4, step=1)
steer_special_tokens = st.sidebar.checkbox("Steer Special Tokens", value=True)

# Chat interface
st.markdown("### Chat Interface")

user_input = st.chat_input("Your Message:", key="user_input")
    
if user_input:
    # Prepare features for the API payload
    features = [
        {
            "modelId": MODEL_ID,
            "layer": feature["layer"],
            "index": feature["index"],
            "strength": feature["strength"],  # Use slider-adjusted strength
        }
        for feature in st.session_state.selected_features
    ]

    # Display the selected features and their context
    # st.markdown("### Selected Features (For Steering)")
    # if features:
    #     for feature in features:
    #         st.markdown(
    #             f"- **Description Context**: `{feature['layer']}:{feature['index']}`<br>"
    #             f"  **Strength**: {feature['strength']}",
    #             unsafe_allow_html=True,
    #         )
    # else:
    #     st.error("No features selected. Steered response may not be influenced.")

    # Prepare API payload
    payload = {
        "defaultChatMessages": [
            {"role": "user", "content": user_input}
        ],
        "steeredChatMessages": [
            {"role": "user", "content": user_input}
        ],
        "modelId": MODEL_ID,
        "features": features,
        "temperature": temperature,
        "n_tokens": n_tokens,
        "freq_penalty": freq_penalty,
        "seed": seed,
        "strength_multiplier": strength_multiplier,
        "steer_special_tokens": steer_special_tokens,
    }

    # Display the full payload being sent
    # st.markdown("### Full API Payload")
    # st.json(payload)

    # API Call and response handling
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        # Parse Default and Steered chat templates
        default_chat = data.get("DEFAULT", {}).get("chat_template", [])
        steered_chat = data.get("STEERED", {}).get("chat_template", [])

        # Extract the latest model response for default and steered
        default_response = (
            default_chat[-1]["content"] if default_chat and default_chat[-1]["role"] == "model" else "No response"
        )
        steered_response = (
            steered_chat[-1]["content"] if steered_chat and steered_chat[-1]["role"] == "model" else "No response"
        )

        # Ensure user sees the steered response in context of features used
        # st.markdown("### Default Model Response")
        # st.write(default_response)

        # st.markdown("### Steered Model Response (Influenced by Features)")
        # if steered_response != "No response" and features:
        #     st.write(f"**Steered response generated using features:**")
        #     for feature in features:
        #         st.markdown(
        #             f"- **Layer**: `{feature['layer']}`<br>"
        #             f"  **Index**: `{feature['index']}`<br>"
        #             f"  **Strength**: {feature['strength']}",
        #             unsafe_allow_html=True,
        #         )
        #     st.write(steered_response)
        # else:
        #     st.warning("Steered response does not appear to be influenced by selected features.")

        # Add user input and responses to memory
        st.session_state.default_memory.chat_memory.add_user_message(user_input)
        st.session_state.default_memory.chat_memory.add_ai_message(default_response)

        st.session_state.steered_memory.chat_memory.add_user_message(user_input)
        st.session_state.steered_memory.chat_memory.add_ai_message(steered_response)

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except (IndexError, TypeError, KeyError) as e:
        st.error(f"Error parsing API response: {e}")

# Display Chat History
col1, col2 = st.columns(2)

# Define background colors
col1_bg_color = '#b5e8c4'  # Color for the first column
col2_bg_color = '#bccfeb'  # Color for the second column

# Display Default Model Chat
with col1:
    default_chat_html = f'''
    <div style="background-color: {col1_bg_color}; padding: 10px; min-height: 70vh; border-radius: 10px">
        <h2>Default Model Chat</h2>
    '''
    for message in st.session_state.default_memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            default_chat_html += f"<p><strong>👤 User:</strong> {message.content}</p>"
        elif isinstance(message, AIMessage):
            default_chat_html += f"<p><strong>🤖 Default Model:</strong> {message.content}</p>"
    default_chat_html += '</div>'
    st.markdown(default_chat_html, unsafe_allow_html=True)

# Display Steered Model Chat
with col2:
    steered_chat_html = f'''
    <div style="background-color: {col2_bg_color}; padding: 10px; min-height: 70vh; border-radius: 10px">
        <h2>Steered Model Chat</h2>
    '''
    for message in st.session_state.steered_memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            steered_chat_html += f"<p><strong>👤 User:</strong> {message.content}</p>"
        elif isinstance(message, AIMessage):
            steered_chat_html += f"<p><strong>🤖 Steered Model:</strong> {message.content}</p>"
    steered_chat_html += '</div>'
    st.markdown(steered_chat_html, unsafe_allow_html=True)
