<template>
  <div class="bg-gray-100 min-h-screen flex flex-col">
    <!-- Header -->
    <header class="bg-blue-600 text-white py-4 shadow-lg">
      <h1 class="text-center text-2xl font-bold">ERP Anomaly Detection</h1>
      <p class="text-center text-sm mt-1">Chat with AI to detect anomalies in ERP transactions</p>
    </header>

    <!-- Chat Container -->
    <main class="flex-1 overflow-y-auto p-4">
      <div class="max-w-3xl mx-auto bg-white shadow-md rounded-lg p-4">
        <!-- Chat Messages -->
        <div class="flex flex-col space-y-4">
          <div
            v-for="(message, index) in messages"
            :key="index"
            :class="[
              'p-3 rounded-lg text-sm',
              message.isUser ? 'bg-blue-100 self-end text-blue-800' : 'bg-gray-200 text-gray-800'
            ]"
          >
            {{ message.text }}
          </div>
        </div>
      </div>
    </main>

    <!-- Input Form -->
    <footer class="bg-gray-200 py-4">
      <div class="max-w-3xl mx-auto flex items-center space-x-4">
        <input
          v-model="userInput"
          class="flex-1 p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring focus:ring-blue-300"
          placeholder="Ask something about ERP anomaly detection..."
        />
        <button
          @click="sendMessage"
          class="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </footer>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      userInput: "",
      messages: [], // History of messages
    };
  },
  methods: {
    async sendMessage() {
      if (!this.userInput.trim()) return;

      // Add user message to chat history
      this.messages.push({ text: this.userInput, isUser: true });

      // Save the prompt
      const prompt = this.userInput;
      this.userInput = ""; // Clear the input field

      try {
        // Send prompt to backend API
        const response = await axios.post("https://<YOUR_NGROK_URL>/chat", {
          prompt,
        });

        // Add AI response to chat history
        this.messages.push({ text: response.data.response, isUser: false });
      } catch (error) {
        console.error("Error:", error);
        this.messages.push({
          text: "Something went wrong. Please try again later.",
          isUser: false,
        });
      }
    },
  },
};
</script>

<style>
/* Optional global styles */
body {
  font-family: 'Inter', sans-serif;
}
</style>
