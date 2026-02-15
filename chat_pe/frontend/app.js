const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

function addMessage(text, sender, images = []) {
  const row = document.createElement("div");
  row.classList.add("message-row", sender);

  const message = document.createElement("div");
  message.classList.add("message", sender);
  message.textContent = text;

  if (images && images.length > 0) {
    const imageContainer = document.createElement("div");
    imageContainer.classList.add("image-container");

    images.forEach((imgData, index) => {
      const wrapper = document.createElement("div");
      wrapper.classList.add("image-wrapper");

      const img = document.createElement("img");
      img.src = imgData;
      img.classList.add("message-image");

      const downloadBtn = document.createElement("a");
      downloadBtn.href = imgData;
      downloadBtn.download = `plot_${Date.now()}_${index}.png`;
      downloadBtn.classList.add("download-btn");
      downloadBtn.innerHTML = "💾"; // Save icon
      downloadBtn.title = "Download Plot";

      wrapper.appendChild(img);
      wrapper.appendChild(downloadBtn);
      imageContainer.appendChild(wrapper);
    });

    message.appendChild(imageContainer);
  }

  row.appendChild(message);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  userInput.value = "";

  try {
    const response = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: text }),
    });

    const data = await response.json();

    if (typeof data.response === 'object') {
      addMessage(data.response.text, "agent", data.response.images);
    } else {
      addMessage(data.response, "agent");
    }

  } catch (err) {
    console.error(err);
    addMessage("⚠️ Could not reach server.", "agent");
  }
});
