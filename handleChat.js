let SESSION_ID = null;
const BASE_URL = "http://localhost:8000/chat";

async function getAnswer(params) {
  let url = "";

  if (SESSION_ID !== null) {
    url = `${BASE_URL}?session_id=${SESSION_ID}`;
  } else {
    url = BASE_URL;
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await response.json();
  return data;
}

async function submitPrompt() {
  document.getElementById("thinking").hidden = false;
  const promptInput = document.querySelector("#prompt-input");
  const userPrompt = promptInput.value;
  const chatHistory = document.getElementById("chat");
  const userPromptElement = document.createElement("p");
  userPromptElement.textContent = userPrompt;
  chatHistory.append(userPromptElement);
  promptInput.value = "";

  const messageObject = { role: "user", content: userPrompt };

  const { answer, session_id } = await getAnswer(messageObject);
  SESSION_ID = session_id;
  const answerElement = document.createElement("p");
  answerElement.textContent = answer;
  chatHistory.append(answerElement);
  document.getElementById("thinking").hidden = true;
}
