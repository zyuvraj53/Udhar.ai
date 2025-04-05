from app import app

if __name__ == '__main__':
    # Replace this:
    # app.run(debug=False, host='0.0.0.0', port=5000)
    
    # With this:
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)