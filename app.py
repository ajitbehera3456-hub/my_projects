from flask import Flask, render_template, request, redirect, session
import pickle
import sqlite3
import requests
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

API_KEY = "ba9af162908053cbe1a5b686ee95a2b5"

# DATABASE
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        news TEXT,
        result TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# EXPLAIN AI
def explain_prediction(text):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    word_importance = vec.toarray()[0] * coef
    top_indices = word_importance.argsort()[-5:][::-1]

    return [feature_names[i] for i in top_indices if word_importance[i] != 0]

# AUTH
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users(username,password) VALUES (?,?)", (u, p))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        user = c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (u, p)
        ).fetchone()
        conn.close()

        if user:
            session["user"] = u
            return redirect("/")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# MAIN
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect("/login")

    prediction = None
    confidence = None
    explanation = []
    articles = []

    if request.method == "POST":
        news = request.form["news"]

        vec = vectorizer.transform([news])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        prediction = "REAL" if pred == 1 else "FAKE"
        confidence = round(max(prob) * 100, 2)

        explanation = explain_prediction(news)

        # SAVE HISTORY
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO history(username, news, result, confidence) VALUES (?,?,?,?)",
            (session["user"], news, prediction, confidence)
        )
        conn.commit()
        conn.close()

        # NEWS API
        try:
            url = f"https://gnews.io/api/v4/search?q={news}&lang=en&country=in&max=6&apikey={API_KEY}"
            response = requests.get(url)

            print("STATUS:", response.status_code)

            if response.status_code == 200:
                data = response.json()

                for a in data.get("articles", []):
                    articles.append({
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "image": a.get("image"),
                        "link": a.get("url")
                    })
            else:
                print("API FAILED:", response.text)

        except Exception as e:
            print("ERROR:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        articles=articles
    )

# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    data = c.execute(
        "SELECT result FROM history WHERE username=?",
        (session["user"],)
    ).fetchall()

    total = len(data)
    real = sum(1 for x in data if x[0] == "REAL")
    fake = sum(1 for x in data if x[0] == "FAKE")

    history = c.execute(
        "SELECT news, result, confidence FROM history WHERE username=?",
        (session["user"],)
    ).fetchall()

    conn.close()

    # TRENDING NEWS
    trending = []
    try:
        url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=6&apikey={API_KEY}"
        response = requests.get(url)

        print("TREND STATUS:", response.status_code)

        if response.status_code == 200:
            data = response.json()

            for a in data.get("articles", []):
                trending.append({
                    "title": a.get("title"),
                    "image": a.get("image"),
                    "link": a.get("url")
                })
        else:
            print("TREND API FAILED:", response.text)

    except Exception as e:
        print("TREND ERROR:", e)

    return render_template(
        "dashboard.html",
        total=total,
        real=real,
        fake=fake,
        history=history,
        trending=trending
    )

if __name__ == "__main__":
    app.run(debug=True)
