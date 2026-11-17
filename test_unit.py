"""Check application behavior with explicit mocked Watson responses."""

import unittest
from unittest.mock import Mock, patch

import requests

from EmotionDetection import emotion_detection, emotion_detector
from server import app


class TestWatsonContract(unittest.TestCase):
    """Validate the HTTP contract and formatted response independently of Watson."""

    def setUp(self):
        self.scores = {
            "anger": 0.08, "disgust": 0.02, "fear": 0.03,
            "joy": 0.91, "sadness": 0.12,
        }

    def response(self, status=200):
        result = Mock(status_code=status)
        result.json.return_value = {
            "emotionPredictions": [{"emotion": self.scores}]
        }
        return result

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_request_contract_and_format(self, post):
        post.return_value = self.response()
        text = "A & B + C? I am happy!"
        result = emotion_detector(text)
        post.assert_called_once_with(
            emotion_detection.URL,
            json={"raw_document": {"text": text}},
            headers={
                "grpc-metadata-mm-model-id":
                "emotion_aggregated-workflow_lang_en_stock"
            },
            timeout=30,
        )
        self.assertEqual(result, {**self.scores, "dominant_emotion": "joy"})

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_emotion_is_highest_score(self, post):
        for expected in self.scores:
            with self.subTest(expected=expected):
                self.scores = dict.fromkeys(self.scores, 0.01)
                self.scores[expected] = 0.95
                post.return_value = self.response()
                self.assertEqual(
                    emotion_detector("Example text")["dominant_emotion"], expected
                )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_blank_input_makes_no_request(self, post):
        for text in ("", "   ", "\n\t", None):
            with self.subTest(text=text):
                result = emotion_detector(text)
                self.assertEqual(len(result), 6)
                self.assertTrue(all(value is None for value in result.values()))
        post.assert_not_called()

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_http_400_returns_none_fields(self, post):
        post.return_value = self.response(400)
        result = emotion_detector("Invalid input")
        self.assertEqual(set(result), {*self.scores, "dominant_emotion"})
        self.assertTrue(all(value is None for value in result.values()))
        post.return_value.json.assert_not_called()

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_network_error_is_not_a_prediction(self, post):
        post.side_effect = requests.Timeout("Test timeout")
        with self.assertRaises(requests.Timeout):
            emotion_detector("Example text")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_http_500_is_not_a_prediction(self, post):
        post.return_value = self.response(500)
        post.return_value.raise_for_status.side_effect = requests.HTTPError()
        with self.assertRaises(requests.HTTPError):
            emotion_detector("Example text")


class TestFlaskApplication(unittest.TestCase):
    """Exercise the real Flask routes with a mocked upstream function."""

    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NLP - Emotion Detection", response.data)

    def test_javascript_is_available(self):
        with self.client.get("/static/mywebscript.js") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"URLSearchParams", response.data)

    @patch("server.emotion_detector")
    def test_empty_and_missing_input(self, detector):
        for query in ({}, {"textToAnalyze": ""}, {"textToAnalyze": "  "}):
            response = self.client.get("/emotionDetector", query_string=query)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "Invalid text! Please try again!")
        detector.assert_not_called()

    @patch("server.emotion_detector")
    def test_successful_response(self, detector):
        detector.return_value = {
            "anger": 0.01, "disgust": 0.02, "fear": 0.03,
            "joy": 0.95, "sadness": 0.04, "dominant_emotion": "joy",
        }
        text = "I am happy & excited + grateful?"
        response = self.client.get(
            "/emotionDetector", query_string={"textToAnalyze": text}
        )
        detector.assert_called_once_with(text)
        self.assertEqual(response.status_code, 200)
        for emotion in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIn(emotion, response.text)
        self.assertIn("The dominant emotion is joy.", response.text)

    @patch("server.emotion_detector")
    def test_none_result(self, detector):
        detector.return_value = {"dominant_emotion": None}
        response = self.client.get(
            "/emotionDetector", query_string={"textToAnalyze": "Invalid"}
        )
        self.assertEqual(response.text, "Invalid text! Please try again!")

    @patch("server.emotion_detector")
    def test_upstream_failure_returns_503(self, detector):
        for error in (requests.Timeout(), ValueError(), KeyError(), IndexError()):
            with self.subTest(error=type(error).__name__):
                detector.side_effect = error
                response = self.client.get(
                    "/emotionDetector", query_string={"textToAnalyze": "Hello"}
                )
                self.assertEqual(response.status_code, 503)
                self.assertIn("service is unavailable", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
