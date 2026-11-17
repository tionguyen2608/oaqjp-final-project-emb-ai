# Final project

Emotion Detector is a Flask web application that uses the Watson NLP
EmotionPredict service to analyze English text. It returns scores for anger,
disgust, fear, joy and sadness, together with the dominant emotion.

Author: Nguyễn Tiến Đạt · HE191693

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

On Windows activate the environment with `.venv\Scripts\activate` instead.
Open http://localhost:5000 and enter a statement.

## Validation

```bash
python -m unittest test_unit -v
python -m unittest test_emotion_detection -v
pylint server.py
```

`test_unit.py` checks formatting, request parameters, HTTP 400 handling and
Flask responses using mocked HTTP responses. It does not validate the accuracy
or availability of the Watson service.

`test_emotion_detection.py` calls the real Watson service for the five statements
required by the assignment. These integration tests require access to the
Skills Network service. A service failure is reported as an error, rather than
replaced with fabricated emotion scores.

Blank input returns `Invalid text! Please try again!`. HTTP 400 from Watson
returns a dictionary containing `None` for every field. Network failures and
invalid upstream responses produce a service-unavailable message in Flask.

## Original repository

This project is forked from
[IBM Developer Skills Network](https://github.com/ibm-developer-skills-network/oaqjp-final-project-emb-ai).
The original Apache 2.0 license is retained.
