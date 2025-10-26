# ===========================================
# 🎬 MOVIE ANALYTICS DASHBOARD (FINAL VERSION)
# ===========================================

# ==== IMPORT LIBRARIES ====
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import InconsistentVersionWarning

# ==== SUPPRESS WARNINGS ====
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==== PAGE CONFIG ====
st.set_page_config(
    page_title="🎬 Movie Analytics Dashboard",
    layout="wide",
    page_icon="🎞️"
)

st.title("🎥 Movie Analytics Dashboard")
st.markdown("Analisis dan prediksi data film menggunakan Machine Learning dan Topic Modeling (LDA).")

# ==== UPLOAD DATASET ====
uploaded_df = st.file_uploader("📤 Upload dataset hasil modeling (movie_model_results.csv)", type="csv")

if uploaded_df:
    df = pd.read_csv(uploaded_df)
    st.success(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ==== TEXT MODEL: TF-IDF + LDA (DILATIH SEKALI) ====
    @st.cache_resource
    def load_text_models(df):
        import os
        if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("lda_model.pkl"):
            tfidf = joblib.load("tfidf_vectorizer.pkl")
            lda = joblib.load("lda_model.pkl")
            st.info("✅ TF-IDF & LDA dimuat dari file (tidak dilatih ulang).")
        else:
            st.warning("⚙️ File model teks belum ditemukan — melatih TF-IDF & LDA...")
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(tfidf_matrix)
            joblib.dump(tfidf, "tfidf_vectorizer.pkl")
            joblib.dump(lda, "lda_model.pkl")
            st.success("✅ Model teks berhasil dilatih & disimpan.")
        return tfidf, lda

    tfidf, lda = load_text_models(df)

    # ==== SIDEBAR FILTER ====
    st.sidebar.header("🎛️ Filter Data")
    selected_genre = st.sidebar.multiselect("Pilih Genre", options=df['main_genre'].unique())
    selected_lang = st.sidebar.multiselect("Pilih Bahasa Asli", options=df['original_language'].unique())
    selected_topic = st.sidebar.multiselect("Pilih Topic (LDA)", options=sorted(df['topic'].dropna().unique().tolist()) if 'topic' in df.columns else [])

    filtered_df = df.copy()
    if selected_genre:
        filtered_df = filtered_df[filtered_df['main_genre'].isin(selected_genre)]
    if selected_lang:
        filtered_df = filtered_df[filtered_df['original_language'].isin(selected_lang)]
    if selected_topic:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topic)]

    st.sidebar.markdown(f"📊 Data terfilter: **{len(filtered_df)} film**")

    # ==== TABS ====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exploratory Data Analysis",
        "🧠 Classification Model",
        "💰 Regression Model",
        "🎬 Film Predictor"
    ])

    # ===========================================
    # TAB 1 — EDA
    # ===========================================
    with tab1:
        st.subheader("📈 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(filtered_df, x="vote_average", nbins=30, title="Distribusi Rating Film", color_discrete_sequence=['#0077b6'])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            genre_count = filtered_df['main_genre'].value_counts().reset_index()
            genre_count.columns = ['Genre', 'Jumlah']
            fig2 = px.bar(genre_count, x='Genre', y='Jumlah', color='Genre', title="Jumlah Film per Genre")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        fig3 = px.scatter(filtered_df, x="budget", y="revenue", color="main_genre",
                          hover_data=["title", "vote_average"],
                          title="💰 Korelasi Budget vs Revenue Film")
        st.plotly_chart(fig3, use_container_width=True)

        if 'topic' in filtered_df.columns:
            st.markdown("---")
            fig4 = px.histogram(filtered_df, x="topic", color="main_genre", nbins=5,
                                title="Distribusi Topik Film Berdasarkan Genre")
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("🧠 Visualisasi Topik Film (WordCloud LDA)")

            # Ambil 10 kata teratas dari setiap topik berdasarkan komponen LDA
            words = tfidf.get_feature_names_out()
            n_topics = lda.components_.shape[0]

            cols = st.columns(2)  # tampilkan 2 kolom per baris
            for i in range(n_topics):
                topic_words = lda.components_[i]
                word_freq = {words[j]: topic_words[j] for j in range(len(words))}
                wc = WordCloud(width=600, height=400, background_color='white', colormap='plasma').generate_from_frequencies(word_freq)

                col = cols[i % 2]  # untuk 2 kolom bergantian
                with col:
                    st.markdown(f"**🎭 Topic {i+1}**")
                    st.image(wc.to_array(), use_container_width=True)

    # ===========================================
    # TAB 2 — CLASSIFICATION MODEL
    # ===========================================
    with tab2:
        st.subheader("🧠 Evaluasi Model Klasifikasi Favorite Movie")
        if 'favorite_movie_pred' in df.columns:
            cm = confusion_matrix(df['favorite_movie'], df['favorite_movie_pred'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(3.5,3))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
            plt.xlabel("Predicted"); plt.ylabel("Actual")
            st.pyplot(fig)

            if 'topic' in df.columns:
                st.markdown("---")
                topic_fav = df.groupby('topic')['favorite_movie_pred'].mean().reset_index()
                fig5 = px.bar(topic_fav, x='topic', y='favorite_movie_pred', title="Persentase Film Favorit per Topik (LDA)", color='favorite_movie_pred')
                st.plotly_chart(fig5, use_container_width=True)

    # ===========================================
    # TAB 3 — REGRESSION MODEL
    # ===========================================
    with tab3:
        st.subheader("💰 Model Prediksi Profit Film")
        if 'profit' in df.columns:
            fig6 = px.scatter(filtered_df, x='budget', y='profit', color='main_genre',
                              hover_data=['title', 'vote_average'],
                              title="Hubungan Budget dan Profit Film")
            st.plotly_chart(fig6, use_container_width=True)

            st.markdown("---")
            fig7 = px.box(filtered_df, x='main_genre', y='profit', color='main_genre', title="Sebaran Profit Berdasarkan Genre")
            st.plotly_chart(fig7, use_container_width=True)

            if 'topic' in df.columns:
                st.markdown("---")
                topic_profit = df.groupby('topic')['profit'].mean().reset_index()
                fig8 = px.bar(topic_profit, x='topic', y='profit', color='profit',
                              title="Rata-Rata Profit Berdasarkan Tema Film (LDA)")
                st.plotly_chart(fig8, use_container_width=True)

    # ===========================================
    # TAB 4 — FILM PREDICTOR
    # ===========================================
    with tab4:
        st.subheader("🎬 Movie Success Predictor")
        st.markdown("Masukkan data film baru untuk memprediksi apakah film ini akan menjadi favorit dan seberapa besar profit-nya.")

        col1, col2 = st.columns(2)
        with col1:
            title_input = st.text_input("Judul Film")
            overview_input = st.text_area("Sinopsis (Overview Film)")
            genre_input = st.selectbox("Genre Utama", df['main_genre'].unique())
            lang_input = st.selectbox("Bahasa Asli", df['original_language'].unique())
        with col2:
            budget_input = st.number_input("Budget (USD)", min_value=0, step=100000)
            runtime_input = st.number_input("Durasi (menit)", min_value=0, step=10)
            popularity_input = st.slider("Popularitas", 0.0, 100.0, 10.0)
            vote_count_input = st.number_input("Perkiraan Jumlah Voting", min_value=0, step=100)

        submit = st.button("🔮 Prediksi Film")

        if submit:
            st.info("🚀 Memulai prediksi...")

            # Load model
            clf = joblib.load("favorite_movie_model.pkl")
            reg = joblib.load("profit_regressor.pkl")

            # Gunakan model TF-IDF & LDA dari atas
            topic_vector = tfidf.transform([overview_input])
            topic_pred = int(np.argmax(lda.transform(topic_vector)))

            # Encoding genre
            le_genre = LabelEncoder()
            le_genre.fit(df['main_genre'])
            genre_encoded = le_genre.transform([genre_input])[0]

            # Buat dataframe input
            duration_hours = runtime_input / 60
            data = pd.DataFrame([{
                'budget': budget_input,
                'revenue': 0,
                'popularity': popularity_input,
                'runtime': runtime_input,
                'title_len': len(title_input),
                'duration_hours': duration_hours,
                'is_english': 1 if lang_input == 'en' else 0,
                'main_genre_encoded': genre_encoded,
                'topic': topic_pred
            }])

            # Prediksi
            fav_pred = clf.predict(data)[0]
            profit_pred = reg.predict(data)[0]

            # Hasil
            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                st.metric("🎯 Prediksi Favorite Movie", "⭐ Favorit!" if fav_pred == 1 else "⚪ Biasa Saja")
            with colB:
                st.metric("💰 Estimasi Profit (USD)", f"${profit_pred:,.0f}")

            if fav_pred == 1:
                st.success("Film ini berpotensi menjadi *favorite* di kalangan penonton berdasarkan pola dataset.")
            else:
                st.warning("Film ini memiliki karakteristik yang mirip dengan film biasa di dataset.")

            st.caption("Prediksi menggunakan model Decision Tree & Random Forest.")

else:
    st.info("⬆️ Silakan upload file terlebih dahulu.")
