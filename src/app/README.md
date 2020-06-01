# CODEtransformer - Web Application

CODEtransformer's web application.

## Run

To run the application in localhost, run the following commands.

1. Create an isolated Python environment and activate it.
```bash
python -m venv env
env\Scripts\activate
```

2. Install the specified requirements.
```bash
pip install -r requirements.txt
```

3. Run the application.
```bash
python main.py
```

4. Open localhost in browser.
```bash
http://localhost:8080
```

## Deploy

To deploy the application on Google Cloud Platform's App Engine, run the following commands.

1. Deploy the web application.
```bash
gcloud app deploy
```

2. Launch the application in browser.
```bash
gcloud app browse
```

**Google Cloud Platform's SDK required.**