# 📡 OpinionLens (Reddit Radar AI)

**Advanced Brand Sentiment & Market Intelligence System**

OpinionLens (aka Reddit Radar AI) is a powerful, web-based intelligence tool that aggregates and analyzes community discussions to provide actionable insights. Built with Python and Streamlit, it evaluates public sentiment on brands, products, and topics using real-time data from Reddit and other sources.

---

## ✨ Features

- **🔍 Search Brand:** Instantly scan Reddit for any brand or product (e.g., "iPhone 16", "GTA VI") to gather community feedback.
- **🔗 Analyze Specific Links:** Deep-dive into specific Reddit post comments or paste text from other platforms (like Instagram) for targeted analysis.
- **🧠 AI Sentiment Analysis:** Automatically categorizes feedback into Positive, Neutral, or Negative sentiments.
- **📊 Buy Confidence Score:** Calculates an overall "Buy Score" and provides a clear AI verdict (e.g., "Highly Recommended", "Mixed Feelings", "Avoid").
- **🔑 Intelligent Reasoning:** Extracts common keywords, themes, and highlights the most helpful community reviews (Pros/Cons).
- **📈 Interactive Dashboards:** Visualizes community mood and volume using modern Plotly charts.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jaiamar/OpinionLens.git
   cd OpinionLens
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install streamlit pandas plotly praw
   # (Add additional dependencies from your environment as needed)
   ```

### Running the Application

Start the Streamlit development server:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - The web framework used for the interactive UI.
- **[Plotly](https://plotly.com/python/)** - For dynamic data visualization.
- **Python** - Core logic, scraping, and sentiment analysis.