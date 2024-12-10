<template>
  <div class="sentiment">
    <h1>IMDB Sentiment Analysis</h1>
    <form @submit.prevent="analyzeSentiment" @keyup.enter="analyzeSentiment">
      <label for="review" id="labalreview">Enter your review:</label>
      <textarea
        id="review"
        v-model="review"
        placeholder="Write your review here..."
        rows="6"
        required
      ></textarea>
      <button type="submit" :disabled="loading">
        {{ loading ? "Analyzing..." : "Analyze Sentiment" }}
      </button>
    </form>

    <div v-if="result" class="result">
      <h2>Prediction Result:</h2>
      <p><strong>Sentiment:</strong> {{ result.sentiment }}</p>
      <p>
        <strong>Probability of positive sentiment:</strong>
        {{ result.positiveProbability.toFixed(2) }}%
      </p>
      <p>
        <strong>Probability of negative sentiment:</strong>
        {{ result.negativeProbability.toFixed(2) }}%
      </p>
    </div>

    <div v-if="error" class="error">
      <p>Error: {{ error }}</p>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "SentimentAnalysis",
  data() {
    return {
      review: "",
      result: null,
      loading: false,
      error: null,
    };
  },
  methods: {
    async analyzeSentiment() {
      this.loading = true;
      this.error = null;
      this.result = null;

      try {
        const response = await axios.post("http://127.0.0.1:8000/predict", {
          review: this.review,
        });

        this.result = {
          sentiment: response.data.sentiment,
          positiveProbability: response.data.probability.positive * 100,
          negativeProbability: response.data.probability.negative * 100,
        };
      } catch (err) {
        this.error = err.response?.data?.detail || "Api not started";
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>

.sentiment {
  max-width: 600px;
  margin: 2rem auto;
  padding: 1rem;
  border: 1px solid #8aa9e8;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  font-family: Arial, sans-serif;
  color: #417abd;
}

form {
  margin-bottom: 1.5rem;
}

textarea {
  width: 100%;
  padding: 0.5vh;
  margin-bottom: 1rem;
  margin-top: 1rem  ;
  border: 1px solid #8aa9e8;
  border-radius: 4px;
  font-size: 1rem;


}

button {
  padding: 0.5rem 1rem;
  background-color: #007bff;
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
}

button:disabled {
  background-color: #d6d6d6;
  cursor: not-allowed;
}

.result {
  background-color: #f8f9fa;
  padding: 1rem;
  border: 1px solid #dee2e6;
  border-radius: 4px;
}

.error {
  color: red;
  font-weight: bold;
}


</style>
