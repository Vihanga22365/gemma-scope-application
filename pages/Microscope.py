import streamlit as st 
import requests
import pandas as pd
import altair as alt
import re  # For tokenization
from streamlit_extras.switch_page_button import switch_page

# Constants
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api/search-all"
MODEL_ID = "gpt2-small"
SOURCE_SET = "res-jb"
SELECTED_LAYERS = ["6-res-jb"]
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": "sk-np-h0ZsR5M1gY0w8al332rJUYa0C8hQL2yUogd5n4Pgvvg0"  # Replace with your actual API token
}

# Initialize Session State
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "available_explanations" not in st.session_state:
    st.session_state.available_explanations = []

# Helper Functions
def tokenize_sentence(sentence):
    """Tokenize the input sentence using regex."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)

def fetch_explanations_for_token(token):
    """Fetch explanations from Neuronpedia API for a given token."""
    payload = {
        "modelId": MODEL_ID,
        "sourceSet": SOURCE_SET,
        "text": token,
        "selectedLayers": SELECTED_LAYERS,
        "sortIndexes": [1],
        "ignoreBos": False,
        "densityThreshold": -1,
        "numResults": 50,
    }
    try:
        response = requests.post(NEURONPEDIA_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result_data = response.json().get("result", [])  # Top-level 'result'
        explanations = []  # To store all explanations
    
        # Traverse the results
        for result in result_data:
            neuron = result.get("neuron", {})  # Get 'neuron' object
            if neuron:
                nested_explanations = neuron.get("explanations", [])  # Access 'explanations'
                if isinstance(nested_explanations, list):  # Ensure it's a list
                    for explanation in nested_explanations:
                        explanations.append({
                            "description": explanation.get("description", "No description available"),
                            "neuron": neuron
                        })
    
        return explanations  # Return all explanations with their associated neuron data
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []
    
    
def fetch_neuron_details(layer, index):
    """Fetch neuron details from the Neuronpedia API."""
    payload = {
        "modelId": MODEL_ID,
        "layer": layer,
        "index": index
    }
    try:
        response = requests.post("https://www.neuronpedia.org/api/neuron", json=payload, headers=HEADERS)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        neuron_details = response.json()  # Parse the API response
        if not neuron_details:
            st.warning("No neuron details found.")
        return neuron_details
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {}

def plot_graph(x_data, y_data, title, x_label="X-axis", y_label="Y-axis"):
    """Generate a histogram for visualization."""
    if not x_data or not y_data:
        st.write(f"No data available for {title}.")
        return None
    chart = alt.Chart(pd.DataFrame({"x": x_data, "y": y_data})).mark_bar(color="#A3E4D7").encode(
        x=alt.X("x:Q", title=x_label, axis=alt.Axis(labelColor="#117864")),
        y=alt.Y("y:Q", title=y_label, axis=alt.Axis(labelColor="#148F77"))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# Streamlit App
st.set_page_config(page_title="Token Feature Analysis", layout="wide", page_icon="🔍")
st.markdown("<h1 style='color:#1F618D;text-align:center;'>Token Feature Analysis Dashboard</h1>", unsafe_allow_html=True)


# if st.sidebar.button('Steer', use_container_width=True):
#     switch_page("Steer")

# Sidebar Input
st.sidebar.markdown("<h3 style='color:#1ABC9C;'>Input Sentence</h3>", unsafe_allow_html=True)
sentence = st.sidebar.text_area("Enter a sentence:")
if st.sidebar.button("Generate Tokens"):
    st.session_state.tokens = tokenize_sentence(sentence)

# Apply Custom CSS for Button Styling
st.markdown(
    """
    <style>
    .stButton > button {
        height: 40px;  /* Uniform button height */
        width: auto;  /* Auto width */
        background-color: #007acc;  /* Blue background */
        color: white;  /* White text */
        border-radius: 5px;  /* Rounded corners */
        border: none;  /* No border */
        font-size: 14px;
        font-weight: bold;
        padding: 0 10px;  /* Padding inside the button */
        margin: 0 5px;  /* Margin for spacing between buttons */
    }
    .stButton > button:hover {
        background-color: #005f99;  /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Tokens and Features
if "tokens" in st.session_state and st.session_state.tokens:
    st.markdown("<h2 style='color:#1F618D;'>Sentence Tokenization</h2>", unsafe_allow_html=True)
    
    # Create a row of buttons with Streamlit's columns
    col_count = len(st.session_state.tokens)
    cols = st.columns([1] * col_count)  # Create as many columns as there are tokens with equal width
    for idx, token in enumerate(st.session_state.tokens):
        with cols[idx]:
            if st.button(token, key=f"token_{idx}"):
                st.session_state.selected_token = token

# Fetch and Display Explanations
if st.session_state.selected_token:
    st.markdown(f"<h3 style='color:#1ABC9C;'>Features for Token: {st.session_state.selected_token}</h3>", unsafe_allow_html=True)
    explanations = fetch_explanations_for_token(st.session_state.selected_token)

    if explanations:
        descriptions = [exp["description"] for exp in explanations]
        selected_description = st.selectbox("Select a Feature Description:", descriptions)

        if selected_description and selected_description != "No description available":
            selected_feature = next((exp for exp in explanations if exp["description"] == selected_description), None)
            if selected_feature:
                neuron_data = selected_feature.get("neuron", {})
                if neuron_data:
                    # Display Neuron Data
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("### Negative Logits")
                        neg_str = neuron_data.get("neg_str", [])
                        neg_values = neuron_data.get("neg_values", [])
                        if neg_str and neg_values:
                            st.dataframe(pd.DataFrame({"Word": neg_str, "Value": neg_values}))
                        else:
                            st.write("No Negative Logits available.")
                    with cols[1]:
                        st.markdown("### Positive Logits")
                        pos_str = neuron_data.get("pos_str", [])
                        pos_values = neuron_data.get("pos_values", [])
                        if pos_str and pos_values:
                            st.dataframe(pd.DataFrame({"Word": pos_str, "Value": pos_values}))
                        else:
                            st.write("No Positive Logits available.")

                    # Render Histograms
                    freq_x = neuron_data.get("freq_hist_data_bar_values", [])
                    freq_y = neuron_data.get("freq_hist_data_bar_heights", [])
                    if freq_x and freq_y:
                        st.markdown("### Frequency Histogram")
                        st.altair_chart(plot_graph(freq_x, freq_y, "Frequency Histogram", "Values", "Frequency"), use_container_width=True)
                    logits_x = neuron_data.get("logits_hist_data_bar_values", [])
                    logits_y = neuron_data.get("logits_hist_data_bar_heights", [])
                    if logits_x and logits_y:
                        st.markdown("### Logits Histogram")
                        st.altair_chart(plot_graph(logits_x, logits_y, "Logits Histogram", "Values", "Logits"), use_container_width=True)
                        
                        
                    feature_layer = neuron_data.get("layer")
                    feature_index = neuron_data.get("index")
                    
                    neuron_activation_details = fetch_neuron_details(feature_layer, feature_index)
                    st.write(f"### Top Activations for Layer: {feature_layer}, Index: {feature_index}")
                    
                    
                    activations = neuron_activation_details.get("activations")
                    all_data = []

                    if activations:
                        for activation in activations:
                            tokens_list = activation.get("tokens", [])
                            values_list = activation.get("values", [])
                            
                            if tokens_list and not isinstance(tokens_list[0], list):
                                tokens_list = [tokens_list]
                            if values_list and not isinstance(values_list[0], list):
                                values_list = [values_list]
                                
                            tokens = [token for sublist in tokens_list for token in sublist]
                            values = [value for sublist in values_list for value in sublist]
                            
                            # Find highest value token
                            max_value = max(values)
                            max_value_index = values.index(max_value)
                            max_value_token = tokens[max_value_index].replace('▁', ' ')
                            
                            # Build HTML with tokens and tooltips
                            html_tokens = []
                            max_value = max(values) if values else 1

                            for token, value in zip(tokens, values):
                                token_text = token.replace('▁', ' ')
                                if value > 0:
                                    # Normalize the value between 0 and 1
                                    normalized = value / max_value
                                    # Calculate green intensity (higher value -> darker green)
                                    green = int(150 + (200 * (1 - normalized)))
                                    
                                    color = f'rgb(0 {green} 0)'
                                    color_style = (
                                        f'background-color: {color}; padding: 2px 6px 6px 2px; '
                                        'border-radius: 10px; margin-left: 3px; margin-right: 3px; color: white'
                                    )
                                else:
                                    color_style = ''
                                html_token = f'<span style="{color_style}" title="{value}">{token_text}</span>'
                                html_tokens.append(html_token)
                                
                            # Join tokens into a sentence
                            sentence = ''.join(html_tokens)
                            
                            # Append data to the list
                            all_data.append({
                                'Top Activation': f"{max_value_token}<br> {max_value:.3f}",
                                'Text': sentence
                            })

                    
                    else:
                        st.write("No activations available.")
                        
                # Create DataFrame with all data
                df = pd.DataFrame(all_data)

                # Display table with HTML and remove index
                st.write(
                    df.style.set_table_styles([
                        {'selector': 'th.col0', 'props': [('width', '20%')]},
                        {'selector': 'th.col1', 'props': [('width', '80%')]},
                        {'selector': 'td.col0', 'props': [('width', '20%')]},
                        {'selector': 'td.col1', 'props': [('width', '80%')]}
                    ]).set_properties(**{
                        'text-align': 'left',
                        'white-space': 'pre-wrap',
                        'width': '100%'
                    }).to_html(index=False, escape=False),
                    unsafe_allow_html=True
                )
    else:
        st.warning("No explanations found for the selected token.")
