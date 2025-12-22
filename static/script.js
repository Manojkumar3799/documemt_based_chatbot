document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let formData = new FormData(this);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("uploadStatus").innerText = data.message;
    });
});

function askQuestion() {
    let question = document.getElementById("question").value;

    fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: question})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("answer").innerText = data.answer;
    });
}
