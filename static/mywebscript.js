async function RunSentimentAnalysis() {
    const text = document.getElementById("textToAnalyze").value;
    const output = document.getElementById("system_response");
    const button = document.getElementById("analyzeButton");
    output.textContent = "Analyzing...";
    button.disabled = true;
    try {
        const response = await fetch(
            "/emotionDetector?" + new URLSearchParams({textToAnalyze: text})
        );
        output.textContent = await response.text();
    } catch (error) {
        output.textContent = "Unable to connect. Please try again!";
    } finally {
        button.disabled = false;
    }
}
