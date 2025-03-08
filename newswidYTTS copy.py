import streamlit as st
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from PIL import Image
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to extract transcript details from YouTube video URL
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Function to extract video ID from YouTube video URL
def get_video_id(video_url):
    # Extract video ID from URL
    video_id = video_url.split("v=")[-1]
    return video_id

# Function to retrieve YouTube comments
def get_youtube_comments(video_id):
    comments = []
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        for segment in transcript:
            comments.append(segment['text'])
    except Exception as e:
        st.error(f"Error retrieving comments: {e}")
    return comments

# Function to perform sentiment analysis
def analyze_sentiment(comments):
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    for comment in comments:
        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            positive_count += 1
        elif polarity < 0:
            negative_count += 1
        else:
            neutral_count += 1
    total_comments = len(comments)
    return {
        "positive_percentage": (positive_count / total_comments) * 100,
        "negative_percentage": (negative_count / total_comments) * 100,
        "neutral_percentage": (neutral_count / total_comments) * 100
    }

# Function to generate summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to perform news sentiment analysis
def fetch_news_articles(country_code):
    # Replace YOUR_API_KEY with your actual NewsAPI key
    API_KEY = '999bb93e080c4b6eb99b0f3e1acf034e'
    url = f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={API_KEY}"

    # Send a request to the NewsAPI to get the top headlines
    response = requests.get(url).json()

    # Check if articles are available in the response
    if 'articles' in response:
        return response['articles']
    else:
        return None

# Function to analyze the sentiment of each article
def analyze_sentiment_article(article):
    text = article['title']
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to summarize articles using Sumy
def summarize_articles(articles):
    high_positive_articles = []
    low_negative_articles = []
    neutral_articles = []

    for article in articles:
        sentiment_score = analyze_sentiment_article(article)
        if sentiment_score > 0.5:
            high_positive_articles.append(article)
        elif sentiment_score < -0.5:
            low_negative_articles.append(article)
        else:
            neutral_articles.append(article)

    return high_positive_articles, low_negative_articles, neutral_articles

# Function to visualize sentiment scores title-wise
def visualize_sentiment_scores(articles):
    # Extract article titles and sentiment scores
    titles = [article['title'] for article in articles]
    sentiment_scores = [analyze_sentiment_article(article) for article in articles]

    # Create a bar plot for sentiment scores title-wise
    plt.figure(figsize=(10, 6))
    sns.barplot(x=titles, y=sentiment_scores, palette="coolwarm")
    plt.xticks(rotation=90)
    plt.xlabel("Article Titles")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Scores of News Article Titles")
    st.pyplot()

# Function to summarize articles using Sumy
def summarize_articles_sumy(articles):
    for article in articles:
        title = article['title']
        content = article['content']

        if content:
            st.write(f"**Title:** {title}")
            st.write("Summary:")
            summarize_content(content)
        else:
            st.warning(f"No content available for the article: {title}")

# Function to summarize content using Sumy
def summarize_content(content):
    parser = PlaintextParser.from_string(content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=2)
    for sentence in summary:
        st.write(sentence)
def summarize_content(content):
    parser = PlaintextParser.from_string(content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=2)
    return summary

# Streamlit web app
def main():
    st.sidebar.title("Document Summarization")

    # Input text area for the document
    document = st.sidebar.text_area("Enter Document Here", height=200)

    # Button to summarize the document
    if st.sidebar.button("Summarize"):
        if document:
            summary = summarize_content(document)
            st.subheader("Summary:")
            for sentence in summary:
                st.write(sentence)
        else:
            st.warning("Please enter some text in the document.")

    # Add the developed by text
    st.sidebar.markdown("<div class='developed-by'>Developed By - SHREE HARI VAITHIYANATHAN</div>", unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.markdown("Here's how it works:")
    st.sidebar.markdown("1. Enter the document in the text area.")
    st.sidebar.markdown("2. Click on the 'Summarize' button.")
    st.sidebar.markdown("3. View the summarized document on the right side.")

    # Custom CSS for the Streamlit app
    custom_css = """
    <style>
    /* Sidebar */
    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
        align-items: flex-start; /* Align sidebar content to the left */
    }

    /* Sidebar Title */
    .sidebar .sidebar-content .title {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px; /* Add margin above the title */
    }

    /* Sidebar Markdown */
    .sidebar .markdown-text-container {
        padding-left: 20px;
    }

    /* Developed By text */
    .developed-by {
        align-self: flex-start; /* Align text to the left */
        margin-top: 10px; /* Add margin above the text */
        margin-left: 0px; /* Add margin to the left of the text */
        font-size: 15px;
        color: #666;
        font-weight: bold; /* Make text bold */
    }
    </style>
    """

    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Main content area
    #st.title("Document Summarization with Sumy")

if __name__ == "__main__":
    main()
# Streamlit web app
def main():
    st.title("YouTube Transcript & News Summarization")
    prompt = """Welcome to the world of infinite knowledge! 
    As a YouTube video summarizer, you possess the extraordinary ability to distill the essence of any video into a 
    concise and insightful summary. Your mission is to unravel the secrets hidden within the transcript, 
    weaving together a narrative that sparks curiosity and ignites the imagination. With your words, 
    you have the power to transport readers on a journey of discovery, revealing profound insights and 
    illuminating the path to enlightenment. Embrace the challenge, embrace the magic, and let your summary 
    shine like a beacon of knowledge in the vast sea of information. The world awaits your brilliance
    dazzle them with your wisdom!"""

    # Input for selecting country code
    country_code = st.sidebar.selectbox("Select Country Code", ["us", "gb", "in", "ca", "au"])

    # Button to fetch news articles and perform analysis
    if st.sidebar.button("Fetch News and Analyze Sentiment"):
        st.write("Fetching news articles...")
        articles = fetch_news_articles(country_code)

        if articles:
            st.write("Analyzing sentiment of news articles...")

            # Visualize sentiment scores title-wise
            visualize_sentiment_scores(articles)

            # Summarize articles based on sentiment
            high_positive_articles, low_negative_articles, neutral_articles = summarize_articles(articles)

            # Summarize high positive sentiment articles
            st.subheader("High Positive Sentiment Articles:")
            summarize_articles_sumy(high_positive_articles)

            # Summarize low negative sentiment articles
            st.subheader("Low Negative Sentiment Articles:")
            summarize_articles_sumy(low_negative_articles)

            # Summarize neutral sentiment articles
            st.subheader("Neutral Sentiment Articles:")
            summarize_articles_sumy(neutral_articles)

        else:
            st.error("No articles found in the response.")

    # Input fields for YouTube link and title
    #@st.title("YouTube Transcript to Detailed Notes Converter")
    youtube_link = st.text_input("Enter YouTube Video Link:")
    file_name = st.text_input("Enter File Name:")

    if youtube_link:
        video_id = get_video_id(youtube_link)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        # Automatically generate detailed notes when a YouTube link is provided
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)

            # Save the detailed notes to a document
            if file_name:
                with open(f"{file_name}.txt", "w") as file:
                    file.write(summary)
                    st.success(f"File '{file_name}.txt' saved successfully!")

        # Perform sentiment analysis
        comments = get_youtube_comments(video_id)
        if comments:
            sentiment_analysis = analyze_sentiment(comments)
            st.write("Sentiment Analysis Results:")
            st.write(f"Positive: {sentiment_analysis['positive_percentage']:.2f}%")
            st.write(f"Negative: {sentiment_analysis['negative_percentage']:.2f}%")
            st.write(f"Neutral: {sentiment_analysis['neutral_percentage']:.2f}%")

            # Visualize sentiment scores
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=["Positive", "Negative", "Neutral"],
                        y=[sentiment_analysis["positive_percentage"],
                           sentiment_analysis["negative_percentage"],
                           sentiment_analysis["neutral_percentage"]],
                        palette="viridis", ax=ax)
            ax.set_xlabel("Sentiment Scores")
            ax.set_ylabel("Percentage")
            ax.set_title("Sentiment Analysis Based On Comments")
            st.pyplot(fig)

# Custom CSS for the Streamlit app
custom_css = """
<style>
/* Body */
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    color: #333;
}

/* Sidebar */
.sidebar .sidebar-content {
    display: flex;
    flex-direction: column;
    align-items: flex-start; /* Align sidebar content to the left */
}

/* Sidebar Title */
.sidebar .sidebar-content .title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px; /* Add margin above the title */
}

/* Sidebar Markdown */
.sidebar .markdown-text-container {
    padding-left: 20px;
}

/* Developed By text */
.developed-by {
    align-self: flex-start; /* Align text to the left */
    margin-top: 10px; /* Add margin above the text */
    margin-left: 0px; /* Add margin to the left of the text */
    font-size: 15px;
    color: #666;
    font-weight: bold; /* Make text bold */
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Add the YouTube icon above the sidebar title
youtube_icon_path = "C:\\Users\\Lenovo\\OneDrive\\Documents\\you_tube_dev\\youtube-logo-png-2069.png"
youtube_icon = Image.open(youtube_icon_path)
#
st.sidebar.image(youtube_icon, width=85, caption='YouTube', use_column_width=False, 
                 output_format='auto')
#st.sidebar.markdown("<div class='developed-by'>Developed By - Sree Hari Vaithiyanathan</div>", unsafe_allow_html=True)
st.sidebar.title("Work Flow Of The Process")

# Sidebar content
st.sidebar.markdown("Here's how it works:")
st.sidebar.markdown("1. Input the YouTube video link you want to transcribe.")
st.sidebar.markdown("2. Select the subject area Like Education, News, Entertainment, Sports, Gaming & Lot More.")
st.sidebar.markdown("3. Click on the 'Get Detailed Notes' & Fetch News And Analyze Sentiment button.")
st.sidebar.markdown("4. Watch in amazement as the AI-powered model generates detailed notes & Sentiment Scores based on the video transcript & Comments!")

if __name__ == "__main__":
    main()
