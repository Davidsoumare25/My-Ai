from flask import Flask, render_template, request, redirect
import sqlite3

app = Flask(__name__)

# Initialisation de la base de données (se lance au démarrage)
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Table pour les utilisateurs
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT)''')
    # Table pour les messages
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, content TEXT, author TEXT)''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT author, content FROM messages ORDER BY id DESC")
    all_messages = cursor.fetchall()
    conn.close()
    return render_template('index.html', messages=all_messages)

@app.route('/send', methods=['POST'])
def send():
    author = request.form.get('username')
    content = request.form.get('message')
    if author and content:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (author, content) VALUES (?, ?)", (author, content))
        conn.commit()
        conn.close()
    return redirect('/')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

