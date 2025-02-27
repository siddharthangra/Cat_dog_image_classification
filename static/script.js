// Typewriter effect
const text = "'Meow-Meow' harked the cat on bed,'Woof-Woof' called the dog wanting to play";
let index = 0;

function typeWriter() {
    if (index < text.length) {
        document.getElementById("typewriter").innerHTML += text.charAt(index);
        index++;
        setTimeout(typeWriter, 50);
    }
}

// Function to close hover box with animation
function closeOverlay() {
    let overlay = document.getElementById("overlay");
    overlay.classList.add("fade-out");

    setTimeout(() => {
        overlay.style.display = "none";
    }, 1500); // Matches animation duration
}

window.onload = function () {
    typeWriter();
};

// Change text when an image is uploaded
document.getElementById("imageUpload").addEventListener("change", function () {
    let fileInput = document.getElementById("imageUpload");
    let file = fileInput.files[0];

    if (file) {
        document.getElementById("predictionResult").innerHTML = "🐾⏳ Waiting for prediction... 🐾";
    }
});

// Image upload function
function uploadImage() {
    let fileInput = document.getElementById("imageUpload");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    let reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("previewImage").src = e.target.result;
        document.getElementById("previewImage").style.display = "block";
    };
    reader.readAsDataURL(file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("predictionResult").innerHTML = "❌ Error: " + data.error;
        } else {
            document.getElementById("predictionResult").innerHTML = "✅ Prediction: <strong>" + data.prediction + "</strong>";
        }
    })
    .catch(error => {
        document.getElementById("predictionResult").innerHTML = "❌ Error: " + error;
    });
}
